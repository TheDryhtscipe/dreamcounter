//! Runtime values (spec §11).
//!
//! Values are immutable and cheap to clone: compound values sit behind `Rc`, and
//! arrays are persistent vectors whose updates take O(log n). Lists cache their
//! length in every cons cell, so `length` is O(1) (spec §5).
//!
//! Nothing here recurses on the host stack over the value structure. Dropping a
//! value frees its uniquely owned parts with an explicit work list (see the `Drop`
//! impls), so dropping a list of a million elements, a deeply nested option or a
//! long chain of closures cannot overflow the Rust stack. Structural equality is
//! iterative too. Only `Display`/`Debug`, which exist for test output, recurse.

use std::cell::RefCell;
use std::fmt;
use std::mem;
use std::rc::Rc;

use dc_core::{Prim, TermId};
use rustc_hash::FxHashMap;

/// A dc value.
#[derive(Clone)]
pub enum Value {
    /// 64-bit signed integer.
    Int(i64),
    /// Boolean.
    Bool(bool),
    /// The unit value.
    Unit,
    /// A pair.
    Pair(Rc<(Value, Value)>),
    /// `none` or `some v`.
    Option(Option<Rc<Value>>),
    /// A list.
    List(List),
    /// A persistent array.
    Array(imbl::Vector<Value>),
    /// A binary tree.
    Tree(Tree),
    /// A function value.
    Fun(Rc<Fun>),
}

/// The function values: closures and the internal callables of primitives.
#[derive(Debug)]
pub enum Fun {
    /// A lambda with its captured environment.
    Closure {
        /// The lambda's body (binds index 0).
        body: TermId,
        /// The environment the lambda was created in.
        env: Env,
    },
    /// A primitive applied to fewer arguments than its arity.
    Partial {
        /// The primitive.
        prim: Prim,
        /// The arguments received so far, in order.
        args: Vec<Value>,
    },
    /// The recursive argument `self` passed by `bfix`. Calling it with `x` is
    /// `bfix fuel f x`.
    BfixSelf {
        /// Remaining depth fuel for the next call.
        fuel: i64,
        /// The functional `F`.
        f: Value,
    },
    /// The recursive argument `self` passed by `memorec`.
    MemoSelf {
        /// The memo table shared by every call in one outermost `memorec`.
        table: Rc<RefCell<MemoTable>>,
        /// The functional `F`.
        f: Value,
    },
}

/// The memo table of one `memorec` evaluation (spec §6).
#[derive(Debug, Default)]
pub struct MemoTable {
    /// Completed results.
    pub entries: FxHashMap<MemoKey, Value>,
    /// How many more misses are allowed before `FuelExhausted`.
    pub misses_left: i64,
}

/// One atom of a [`MemoKey`]'s pre-order encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeyAtom {
    /// An integer.
    Int(i64),
    /// A boolean.
    Bool(bool),
    /// The unit value.
    Unit,
    /// A pair; its two components follow in pre-order.
    Pair,
}

/// A memo-table key: `Int`, `Bool`, `Unit` or pairs of those (spec §6).
///
/// Stored as the pre-order sequence of its value nodes. Every atom has a fixed
/// arity, so the encoding is injective, and the key size of the cost model is the
/// length of the sequence.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MemoKey(Vec<KeyAtom>);

impl MemoKey {
    /// Converts a value to a key. Returns `None` for a value that is not a valid key.
    pub fn from_value(v: &Value) -> Option<MemoKey> {
        let mut atoms = Vec::new();
        let mut work = vec![v];
        while let Some(v) = work.pop() {
            match v {
                Value::Int(n) => atoms.push(KeyAtom::Int(*n)),
                Value::Bool(b) => atoms.push(KeyAtom::Bool(*b)),
                Value::Unit => atoms.push(KeyAtom::Unit),
                Value::Pair(p) => {
                    atoms.push(KeyAtom::Pair);
                    work.push(&p.1);
                    work.push(&p.0);
                }
                _ => return None,
            }
        }
        Some(MemoKey(atoms))
    }

    /// The key size in value nodes, as the cost model counts it.
    pub fn size(&self) -> u64 {
        self.0.len() as u64
    }
}

/// A persistent environment: a linked list of frames, innermost binder first.
#[derive(Clone, Debug, Default)]
pub struct Env(Option<Rc<EnvNode>>);

/// One environment frame.
#[derive(Debug)]
pub struct EnvNode {
    value: Value,
    next: Env,
}

impl Env {
    /// The empty environment.
    pub fn empty() -> Env {
        Env(None)
    }

    /// This environment with `value` bound at index 0.
    pub fn push(&self, value: Value) -> Env {
        Env(Some(Rc::new(EnvNode {
            value,
            next: self.clone(),
        })))
    }

    /// The value at de Bruijn index `i`, if bound.
    pub fn lookup(&self, mut i: u32) -> Option<&Value> {
        let mut node = self.0.as_deref()?;
        while i > 0 {
            node = node.next.0.as_deref()?;
            i -= 1;
        }
        Some(&node.value)
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        let mut next = self.0.take();
        while let Some(rc) = next {
            match Rc::try_unwrap(rc) {
                Ok(mut node) => next = node.next.0.take(),
                Err(_) => break,
            }
        }
    }
}

/// A list: `None` is `nil`.
#[derive(Clone, Default)]
pub struct List(pub Option<Rc<Cons>>);

/// A cons cell with the cached length of the list it heads.
pub struct Cons {
    /// The first element.
    pub head: Value,
    /// The rest of the list.
    pub tail: List,
    /// The length of this list (at least 1).
    pub len: u64,
}

impl List {
    /// The empty list.
    pub fn nil() -> List {
        List(None)
    }

    /// `head :: tail`.
    pub fn cons(head: Value, tail: List) -> List {
        let len = tail.len() + 1;
        List(Some(Rc::new(Cons { head, tail, len })))
    }

    /// The number of elements, in O(1).
    pub fn len(&self) -> u64 {
        self.0.as_ref().map_or(0, |c| c.len)
    }

    /// Whether the list is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    /// The elements, front to back.
    pub fn iter(&self) -> ListIter<'_> {
        ListIter(self.0.as_deref())
    }

    /// Builds a list from elements in order.
    pub fn from_values<I>(items: I) -> List
    where
        I: IntoIterator<Item = Value>,
        I::IntoIter: DoubleEndedIterator,
    {
        items
            .into_iter()
            .rev()
            .fold(List::nil(), |tail, v| List::cons(v, tail))
    }
}

/// Iterator over list elements.
#[derive(Debug)]
pub struct ListIter<'a>(Option<&'a Cons>);

impl<'a> Iterator for ListIter<'a> {
    type Item = &'a Value;

    fn next(&mut self) -> Option<&'a Value> {
        let cell = self.0?;
        self.0 = cell.tail.0.as_deref();
        Some(&cell.head)
    }
}

impl Drop for List {
    fn drop(&mut self) {
        let mut next = self.0.take();
        while let Some(rc) = next {
            match Rc::try_unwrap(rc) {
                Ok(mut cell) => next = cell.tail.0.take(),
                Err(_) => break,
            }
        }
    }
}

/// A binary tree: `None` is `leaf`.
#[derive(Clone, Default)]
pub struct Tree(pub Option<Rc<TreeNode>>);

/// An internal tree node.
pub struct TreeNode {
    /// Left subtree.
    pub left: Tree,
    /// The value at this node.
    pub value: Value,
    /// Right subtree.
    pub right: Tree,
}

impl Tree {
    /// The empty tree.
    pub fn leaf() -> Tree {
        Tree(None)
    }

    /// `node left value right`.
    pub fn node(left: Tree, value: Value, right: Tree) -> Tree {
        Tree(Some(Rc::new(TreeNode { left, value, right })))
    }

    /// The number of internal nodes. Iterative, so safe on degenerate trees.
    pub fn size(&self) -> u64 {
        let mut n = 0;
        let mut work = vec![self];
        while let Some(t) = work.pop() {
            if let Some(node) = &t.0 {
                n += 1;
                work.push(&node.left);
                work.push(&node.right);
            }
        }
        n
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        let mut stack: Vec<Rc<TreeNode>> = self.0.take().into_iter().collect();
        while let Some(rc) = stack.pop() {
            if let Ok(mut node) = Rc::try_unwrap(rc) {
                stack.extend(node.left.0.take());
                stack.extend(node.right.0.take());
            }
        }
    }
}

/// Moves the uniquely owned children of `v` onto `out`, leaving `v` with nothing
/// that a recursive drop would descend into. Shared children are left alone: dropping
/// them only decrements a count.
fn detach_children(v: &mut Value, out: &mut Vec<Value>) {
    match v {
        Value::Int(_) | Value::Bool(_) | Value::Unit | Value::Option(None) => {}
        // Array elements are dropped by `imbl` one at a time, and each element's own
        // drop is iterative, so arrays only nest as deep as the program nests them.
        Value::Array(_) => {}
        Value::Pair(rc) => {
            if let Some(p) = Rc::get_mut(rc) {
                out.push(mem::replace(&mut p.0, Value::Unit));
                out.push(mem::replace(&mut p.1, Value::Unit));
            }
        }
        Value::Option(Some(rc)) => {
            if let Some(x) = Rc::get_mut(rc) {
                out.push(mem::replace(x, Value::Unit));
            }
        }
        Value::List(l) => {
            let mut next = l.0.take();
            while let Some(rc) = next {
                match Rc::try_unwrap(rc) {
                    Ok(mut cell) => {
                        out.push(mem::replace(&mut cell.head, Value::Unit));
                        next = cell.tail.0.take();
                    }
                    Err(_) => break,
                }
            }
        }
        Value::Tree(t) => {
            let mut stack: Vec<Rc<TreeNode>> = t.0.take().into_iter().collect();
            while let Some(rc) = stack.pop() {
                if let Ok(mut node) = Rc::try_unwrap(rc) {
                    out.push(mem::replace(&mut node.value, Value::Unit));
                    stack.extend(node.left.0.take());
                    stack.extend(node.right.0.take());
                }
            }
        }
        Value::Fun(rc) => {
            if let Some(f) = Rc::get_mut(rc) {
                match f {
                    Fun::Closure { env, .. } => detach_env(env, out),
                    Fun::Partial { args, .. } => out.append(args),
                    Fun::BfixSelf { f, .. } => out.push(mem::replace(f, Value::Unit)),
                    Fun::MemoSelf { table, f } => {
                        out.push(mem::replace(f, Value::Unit));
                        if let Some(cell) = Rc::get_mut(table) {
                            out.extend(cell.get_mut().entries.drain().map(|(_, v)| v));
                        }
                    }
                }
            }
        }
    }
}

fn detach_env(env: &mut Env, out: &mut Vec<Value>) {
    let mut next = env.0.take();
    while let Some(rc) = next {
        match Rc::try_unwrap(rc) {
            Ok(mut node) => {
                out.push(mem::replace(&mut node.value, Value::Unit));
                next = node.next.0.take();
            }
            Err(_) => break,
        }
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        let mut work = Vec::new();
        detach_children(self, &mut work);
        while let Some(mut v) = work.pop() {
            detach_children(&mut v, &mut work);
            // `v` now owns nothing unique, so its own drop does not recurse.
        }
    }
}

impl Value {
    /// A list value from elements in order.
    pub fn list<I>(items: I) -> Value
    where
        I: IntoIterator<Item = Value>,
        I::IntoIter: DoubleEndedIterator,
    {
        Value::List(List::from_values(items))
    }

    /// A list of integers.
    pub fn int_list(items: &[i64]) -> Value {
        Value::list(items.iter().map(|&n| Value::Int(n)))
    }

    /// A pair value.
    pub fn pair(a: Value, b: Value) -> Value {
        Value::Pair(Rc::new((a, b)))
    }

    /// `some v`.
    pub fn some(v: Value) -> Value {
        Value::Option(Some(Rc::new(v)))
    }

    /// `none`.
    pub fn none() -> Value {
        Value::Option(None)
    }

    /// An array value.
    pub fn array<I: IntoIterator<Item = Value>>(items: I) -> Value {
        Value::Array(items.into_iter().collect())
    }

    /// A tree value.
    pub fn tree(t: Tree) -> Value {
        Value::Tree(t)
    }

    /// Whether this is a function value.
    pub fn is_fun(&self) -> bool {
        matches!(self, Value::Fun(_))
    }

    /// The integer, if this is one.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(n) => Some(*n),
            _ => None,
        }
    }
}

/// The result of comparing two values with `eq`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EqResult {
    /// The values are equal (`true`) or differ (`false`).
    Data(bool),
    /// A function value was reached.
    Function,
}

/// Structural equality of data values, as `eq` computes it, with the number of
/// value-node pairs compared. The comparison runs left to right in pre-order and
/// stops at the first difference or at the first function value; the pair that
/// stopped it is counted.
///
/// Nodes: every scalar, pair, option, array, list cell (including the final `nil`)
/// and tree node (including each `leaf`) is one node. A list or tree value is not a
/// node of its own apart from its cells.
pub(crate) fn structural_eq(a: &Value, b: &Value) -> (EqResult, u64) {
    enum Work<'a> {
        Value(&'a Value, &'a Value),
        List(&'a List, &'a List),
        Tree(&'a Tree, &'a Tree),
    }
    let mut work = vec![Work::Value(a, b)];
    let mut nodes = 0u64;
    while let Some(item) = work.pop() {
        let same = match item {
            Work::List(l, m) => {
                nodes += 1;
                match (&l.0, &m.0) {
                    (None, None) => true,
                    (Some(c), Some(d)) => {
                        work.push(Work::List(&c.tail, &d.tail));
                        work.push(Work::Value(&c.head, &d.head));
                        true
                    }
                    _ => false,
                }
            }
            Work::Tree(s, t) => {
                nodes += 1;
                match (&s.0, &t.0) {
                    (None, None) => true,
                    (Some(m), Some(n)) => {
                        work.push(Work::Tree(&m.right, &n.right));
                        work.push(Work::Value(&m.value, &n.value));
                        work.push(Work::Tree(&m.left, &n.left));
                        true
                    }
                    _ => false,
                }
            }
            Work::Value(Value::List(l), Value::List(m)) => {
                work.push(Work::List(l, m));
                true
            }
            Work::Value(Value::Tree(s), Value::Tree(t)) => {
                work.push(Work::Tree(s, t));
                true
            }
            Work::Value(x, y) => {
                nodes += 1;
                match (x, y) {
                    (Value::Fun(_), _) | (_, Value::Fun(_)) => return (EqResult::Function, nodes),
                    (Value::Int(m), Value::Int(n)) => m == n,
                    (Value::Bool(m), Value::Bool(n)) => m == n,
                    (Value::Unit, Value::Unit) => true,
                    (Value::Pair(p), Value::Pair(q)) => {
                        work.push(Work::Value(&p.1, &q.1));
                        work.push(Work::Value(&p.0, &q.0));
                        true
                    }
                    (Value::Option(None), Value::Option(None)) => true,
                    (Value::Option(Some(p)), Value::Option(Some(q))) => {
                        work.push(Work::Value(p, q));
                        true
                    }
                    (Value::Array(u), Value::Array(v)) => {
                        if u.len() == v.len() {
                            for (p, q) in u.iter().zip(v.iter()).rev() {
                                work.push(Work::Value(p, q));
                            }
                            true
                        } else {
                            false
                        }
                    }
                    _ => false,
                }
            }
        };
        if !same {
            return (EqResult::Data(false), nodes);
        }
    }
    (EqResult::Data(true), nodes)
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Debug for List {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl fmt::Debug for Cons {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cons")
            .field("head", &self.head)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for TreeNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TreeNode")
            .field("value", &self.value)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for Tree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&Value::Tree(self.clone()), f)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{n}"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Unit => f.write_str("()"),
            Value::Pair(p) => write!(f, "({}, {})", p.0, p.1),
            Value::Option(None) => f.write_str("None"),
            Value::Option(Some(v)) => write!(f, "Some ({v})"),
            Value::List(l) => {
                f.write_str("[")?;
                for (i, v) in l.iter().enumerate() {
                    if i > 0 {
                        f.write_str("; ")?;
                    }
                    write!(f, "{v}")?;
                }
                f.write_str("]")
            }
            Value::Array(a) => {
                f.write_str("[|")?;
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        f.write_str("; ")?;
                    }
                    write!(f, "{v}")?;
                }
                f.write_str("|]")
            }
            Value::Tree(t) => match &t.0 {
                None => f.write_str("Leaf"),
                Some(n) => write!(
                    f,
                    "Node ({}, {}, {})",
                    Value::Tree(n.left.clone()),
                    n.value,
                    Value::Tree(n.right.clone())
                ),
            },
            Value::Fun(_) => f.write_str("<fun>"),
        }
    }
}

/// Structural equality for tests and tools. Function values are never equal, even
/// to themselves.
impl PartialEq for Value {
    fn eq(&self, other: &Value) -> bool {
        matches!(structural_eq(self, other).0, EqResult::Data(true))
    }
}
