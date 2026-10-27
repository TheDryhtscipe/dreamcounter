//! Runtime values (spec §11).
//!
//! Values are immutable and cheap to clone: compound values sit behind `Rc`, and
//! arrays are persistent vectors whose updates take O(log n). Lists cache their
//! length in every cons cell, so `length` is O(1) (spec §5).
//!
//! Nothing here recurses on the host stack in proportion to the depth of a value.
//! Dropping a value frees its uniquely owned parts with an explicit work list (see
//! the `Drop` impls); arrays, whose elements `imbl` drops itself, are handed to a
//! per-thread queue that the outermost drop empties. So dropping a list of a
//! million elements, deeply nested options, pairs or arrays, or a long chain of
//! closures or environment frames does not overflow the Rust stack. Structural
//! equality, memo keys, `Display` and `Debug` are iterative too. The remaining host
//! recursion is inside `imbl`'s own tree, whose depth is logarithmic in the size of
//! one array.

use std::cell::{Cell, RefCell};
use std::fmt;
use std::mem;
use std::rc::{Rc, Weak};

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
        /// The memo table shared by every call in one outermost `memorec`. A
        /// machine frame owns the table until that call returns; `self` only
        /// refers to it, so a `self` that escapes cannot keep the table alive
        /// through a cycle (table, entry, `self`, table). See [`crate::cost`].
        table: Weak<RefCell<MemoTable>>,
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

/// The result of [`MemoKey::build_within`].
pub(crate) enum KeyBuild {
    /// The key, of at most the limit's size.
    Key(MemoKey),
    /// The value is not a valid key.
    Invalid,
    /// The key has more nodes than the limit; nothing was kept.
    OverLimit,
}

impl MemoKey {
    /// Converts a value to a key. Returns `None` for a value that is not a valid key.
    pub fn from_value(v: &Value) -> Option<MemoKey> {
        match MemoKey::build_within(v, u64::MAX) {
            KeyBuild::Key(k) => Some(k),
            KeyBuild::Invalid | KeyBuild::OverLimit => None,
        }
    }

    /// Converts a value to a key of at most `limit` nodes. Stops as soon as the
    /// key is invalid or has more than `limit` nodes, so the work done is bounded
    /// by `limit` even for a key whose pairs share components (2^n nodes from n
    /// distinct pairs).
    pub(crate) fn build_within(v: &Value, limit: u64) -> KeyBuild {
        let mut atoms = Vec::new();
        let mut work = vec![v];
        while let Some(v) = work.pop() {
            if atoms.len() as u64 >= limit {
                return KeyBuild::OverLimit;
            }
            match v {
                Value::Int(n) => atoms.push(KeyAtom::Int(*n)),
                Value::Bool(b) => atoms.push(KeyAtom::Bool(*b)),
                Value::Unit => atoms.push(KeyAtom::Unit),
                Value::Pair(p) => {
                    atoms.push(KeyAtom::Pair);
                    work.push(&p.1);
                    work.push(&p.0);
                }
                _ => return KeyBuild::Invalid,
            }
        }
        KeyBuild::Key(MemoKey(atoms))
    }

    /// The key size of `v` in value nodes (saturating), or `None` if `v` is not a
    /// valid key. Each distinct pair is measured once, so the work is linear in the
    /// number of distinct pairs, not in the size, which sharing can make
    /// exponentially larger.
    pub(crate) fn size_of(v: &Value) -> Option<u64> {
        let mut sizes: FxHashMap<*const (Value, Value), u64> = FxHashMap::default();
        // `(value, children measured)`; `results` holds the measured sizes.
        let mut work: Vec<(&Value, bool)> = vec![(v, false)];
        let mut results: Vec<u64> = Vec::new();
        while let Some((v, measured)) = work.pop() {
            match v {
                Value::Int(_) | Value::Bool(_) | Value::Unit => results.push(1),
                Value::Pair(p) => {
                    let ptr = Rc::as_ptr(p);
                    if measured {
                        let b = results.pop()?;
                        let a = results.pop()?;
                        let s = a.saturating_add(b).saturating_add(1);
                        sizes.insert(ptr, s);
                        results.push(s);
                    } else if let Some(&s) = sizes.get(&ptr) {
                        results.push(s);
                    } else {
                        work.push((v, true));
                        work.push((&p.1, false));
                        work.push((&p.0, false));
                    }
                }
                _ => return None,
            }
        }
        results.pop()
    }

    /// Whether `v` converts to this key. The work is bounded by this key's size.
    pub(crate) fn matches(&self, v: &Value) -> bool {
        let mut atoms = self.0.iter();
        let mut work = vec![v];
        while let Some(v) = work.pop() {
            let same = match (atoms.next(), v) {
                (Some(KeyAtom::Int(m)), Value::Int(n)) => m == n,
                (Some(KeyAtom::Bool(m)), Value::Bool(n)) => m == n,
                (Some(KeyAtom::Unit), Value::Unit) => true,
                (Some(KeyAtom::Pair), Value::Pair(p)) => {
                    work.push(&p.1);
                    work.push(&p.0);
                    true
                }
                _ => false,
            };
            if !same {
                return false;
            }
        }
        atoms.next().is_none()
    }

    /// The key size in value nodes, as the cost model counts it.
    pub fn size(&self) -> u64 {
        self.0.len() as u64
    }
}

/// A persistent environment: a linked list of frames, innermost binder first.
#[derive(Clone, Default)]
pub struct Env(Option<Rc<EnvNode>>);

/// One environment frame.
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

    /// The bound values, innermost first.
    pub fn values(&self) -> impl Iterator<Item = &Value> {
        let mut node = self.0.as_deref();
        std::iter::from_fn(move || {
            let n = node?;
            node = n.next.0.as_deref();
            Some(&n.value)
        })
    }
}

impl fmt::Debug for Env {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.values()).finish()
    }
}

impl fmt::Debug for EnvNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EnvNode")
            .field("value", &self.value)
            .finish_non_exhaustive()
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
        // `imbl` drops its elements itself, which would recurse once per level of
        // array nesting. Instead the vector is queued and dropped by the outermost
        // drop on this thread. Dropping a shared vector there only decrements a
        // count, so this costs O(1) per array, not O(size).
        Value::Array(a) => defer_array_drop(mem::take(a)),
        Value::Pair(rc) => {
            if let Some(p) = Rc::get_mut(rc) {
                take_child(&mut p.0, out);
                take_child(&mut p.1, out);
            }
        }
        Value::Option(Some(rc)) => {
            if let Some(x) = Rc::get_mut(rc) {
                take_child(x, out);
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
                    Fun::MemoSelf { f, .. } => out.push(mem::replace(f, Value::Unit)),
                }
            }
        }
    }
}

/// Moves `child` onto `out` unless it is a scalar, which drops without work. So a
/// pair or option emptied by an earlier detach adds nothing to the work list.
fn take_child(child: &mut Value, out: &mut Vec<Value>) {
    if !matches!(
        child,
        Value::Int(_) | Value::Bool(_) | Value::Unit | Value::Option(None)
    ) {
        out.push(mem::replace(child, Value::Unit));
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

thread_local! {
    /// Arrays waiting to be dropped by the outermost drop on this thread.
    static ARRAY_DROPS: RefCell<Vec<imbl::Vector<Value>>> = const { RefCell::new(Vec::new()) };
    /// Whether a drop on this thread is already emptying `ARRAY_DROPS`.
    static DRAINING: Cell<bool> = const { Cell::new(false) };
}

fn defer_array_drop(a: imbl::Vector<Value>) {
    if a.is_empty() {
        return;
    }
    let mut slot = Some(a);
    // During thread teardown the queue may be gone; the array then drops in place
    // when `slot` goes out of scope.
    let _ = ARRAY_DROPS.try_with(|q| {
        if let Some(a) = slot.take() {
            q.borrow_mut().push(a);
        }
    });
}

/// Drops queued arrays until none are left, unless an outer drop on this thread is
/// already doing so. Dropping an array drops its elements, which may queue more
/// arrays; the loop picks those up instead of recursing.
fn drain_array_drops() {
    drain_queue(drop);
}

/// Empties `ARRAY_DROPS`, passing each array to `drop_one`.
fn drain_queue(mut drop_one: impl FnMut(imbl::Vector<Value>)) {
    /// Clears `DRAINING` when the drain ends, also if a drop unwinds, so a later
    /// drop on this thread empties the queue again.
    struct Draining;
    impl Drop for Draining {
        fn drop(&mut self) {
            let _ = DRAINING.try_with(|d| d.set(false));
        }
    }
    let Ok(false) = DRAINING.try_with(|d| d.replace(true)) else {
        return;
    };
    let _draining = Draining;
    loop {
        // The borrow ends before the array drops, so nested drops can queue more.
        let next = ARRAY_DROPS
            .try_with(|q| q.borrow_mut().pop())
            .ok()
            .flatten();
        match next {
            Some(a) => drop_one(a),
            None => break,
        }
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        // A value with no children owns nothing to detach and queues no array.
        if matches!(
            self,
            Value::Int(_) | Value::Bool(_) | Value::Unit | Value::Option(None)
        ) {
            return;
        }
        let mut work = Vec::new();
        detach_children(self, &mut work);
        while let Some(mut v) = work.pop() {
            detach_children(&mut v, &mut work);
            // `v` now owns nothing unique, so its own drop does not recurse.
        }
        drain_array_drops();
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
    /// More than the limit of node pairs were compared; the count is the limit + 1.
    OverLimit,
}

/// Structural equality of data values, as `eq` computes it, with the number of
/// value-node pairs compared. The comparison runs left to right in pre-order and
/// stops at the first difference or at the first function value; the pair that
/// stopped it is counted. It also stops once it has compared more than `limit`
/// pairs, so the work is bounded by `limit` even when sharing makes the values
/// exponentially larger than the memory they use.
///
/// Nodes: every scalar, pair, option, array, list cell (including the final `nil`)
/// and tree node (including each `leaf`) is one node. A list or tree value is not a
/// node of its own apart from its cells.
pub(crate) fn structural_eq(a: &Value, b: &Value, limit: u64) -> (EqResult, u64) {
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
        if nodes > limit {
            return (EqResult::OverLimit, nodes);
        }
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
        write_value(Piece::Tree(self), f)
    }
}

/// A piece of output still to write.
enum Piece<'a> {
    Value(&'a Value),
    Tree(&'a Tree),
    Text(&'static str),
}

/// Writes a value with an explicit work stack, so deeply nested values format
/// without host recursion.
fn write_value(start: Piece<'_>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let mut work = vec![start];
    while let Some(piece) = work.pop() {
        match piece {
            Piece::Text(s) => f.write_str(s)?,
            Piece::Tree(t) => match &t.0 {
                None => f.write_str("Leaf")?,
                Some(n) => {
                    // Pushed in reverse: `Node (left, value, right)`.
                    work.push(Piece::Text(")"));
                    work.push(Piece::Tree(&n.right));
                    work.push(Piece::Text(", "));
                    work.push(Piece::Value(&n.value));
                    work.push(Piece::Text(", "));
                    work.push(Piece::Tree(&n.left));
                    work.push(Piece::Text("Node ("));
                }
            },
            Piece::Value(v) => match v {
                Value::Int(n) => write!(f, "{n}")?,
                Value::Bool(b) => write!(f, "{b}")?,
                Value::Unit => f.write_str("()")?,
                Value::Pair(p) => {
                    work.push(Piece::Text(")"));
                    work.push(Piece::Value(&p.1));
                    work.push(Piece::Text(", "));
                    work.push(Piece::Value(&p.0));
                    work.push(Piece::Text("("));
                }
                Value::Option(None) => f.write_str("None")?,
                Value::Option(Some(x)) => {
                    work.push(Piece::Text(")"));
                    work.push(Piece::Value(x));
                    work.push(Piece::Text("Some ("));
                }
                Value::List(l) => {
                    let items: Vec<&Value> = l.iter().collect();
                    push_seq(&mut work, "[", &items, "]");
                }
                Value::Array(a) => {
                    let items: Vec<&Value> = a.iter().collect();
                    push_seq(&mut work, "[|", &items, "|]");
                }
                Value::Tree(t) => work.push(Piece::Tree(t)),
                Value::Fun(_) => f.write_str("<fun>")?,
            },
        }
    }
    Ok(())
}

/// Pushes `open item; item; ... close` in reverse onto the work stack.
fn push_seq<'a>(
    work: &mut Vec<Piece<'a>>,
    open: &'static str,
    items: &[&'a Value],
    close: &'static str,
) {
    work.push(Piece::Text(close));
    for (i, &x) in items.iter().enumerate().rev() {
        work.push(Piece::Value(x));
        if i > 0 {
            work.push(Piece::Text("; "));
        }
    }
    work.push(Piece::Text(open));
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_value(Piece::Value(self), f)
    }
}

/// Structural equality for tests and tools. Function values are never equal, even
/// to themselves.
impl PartialEq for Value {
    fn eq(&self, other: &Value) -> bool {
        matches!(structural_eq(self, other, u64::MAX).0, EqResult::Data(true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn queued() -> usize {
        ARRAY_DROPS.with(|q| q.borrow().len())
    }

    #[test]
    fn a_panic_while_draining_does_not_stop_later_drains() {
        std::thread::spawn(|| {
            defer_array_drop(imbl::vector![Value::Int(1)]);
            defer_array_drop(imbl::vector![Value::Int(2)]);
            let r = std::panic::catch_unwind(|| drain_queue(|_| panic!("drop failed")));
            assert!(r.is_err());
            assert!(!DRAINING.with(Cell::get), "the flag is cleared on unwind");
            assert_eq!(queued(), 1);
            // The next outermost drop empties what the failed one left.
            drop(Value::some(Value::Int(0)));
            assert_eq!(queued(), 0);
        })
        .join()
        .expect("test thread");
    }
}
