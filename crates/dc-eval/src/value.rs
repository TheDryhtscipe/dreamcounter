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
//!
//! Values can share parts, so a value can be exponentially larger as a tree than
//! in memory: `loop 64 (node $0 i $0)` has 2^64 nodes. `PartialEq` remembers the
//! pairs of shared parts it has compared, so its work is bounded by the memory
//! the two values use, not by their size as trees. `Debug` and
//! [`Value::display_within`] write a bounded prefix and mark the rest with `...`;
//! `Display` writes the whole tree, so use it only on values known to be small.

use std::cell::{Cell, RefCell};
use std::fmt;
use std::mem;
use std::ptr;
use std::rc::{Rc, Weak};

use dc_core::{Prim, TermId};
use rustc_hash::{FxHashMap, FxHashSet};

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
        /// The memo table shared by every call in one `memorec`. The machine owns
        /// the table until the evaluation ends; `self` only refers to it, so a
        /// `self` stored in the table cannot keep it alive through a cycle (table,
        /// entry, `self`, table) after that. See [`crate::cost`].
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

/// One atom of a small [`MemoKey`]'s pre-order encoding.
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

/// One distinct subvalue of a large [`MemoKey`]: a scalar, or a pair of earlier
/// nodes given by their positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum KeyNode {
    Int(i64),
    Bool(bool),
    Unit,
    Pair(usize, usize),
}

/// The stored form of a [`MemoKey`]. Which form a key takes depends only on its
/// size, so two equal keys always have the same form.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum KeyRepr {
    /// The pre-order atoms of a key of at most [`FLAT_KEY_MAX`] nodes. Every atom
    /// has a fixed arity, so the encoding is injective.
    Flat(Vec<KeyAtom>),
    /// The distinct subvalues of a larger key, each once, in the order a
    /// left-to-right post-order walk of the key as a tree first completes them;
    /// the last is the key. Equal subvalues share a node, so the sequence depends
    /// only on the key as a tree, and its length on the distinct subvalues.
    Dag(Vec<KeyNode>),
}

/// The largest key, in value nodes, stored in the flat form.
const FLAT_KEY_MAX: u64 = 64;

/// A memo-table key: `Int`, `Bool`, `Unit` or pairs of those (spec §6).
///
/// Small keys are stored as the pre-order sequence of their value nodes. Larger
/// keys are stored with each distinct subvalue once, so a key whose pairs share
/// components (2^n nodes from n distinct pairs) takes memory and work in
/// proportion to its distinct pairs. The key size of the cost model is the size
/// of the key as a tree, whichever form it takes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MemoKey {
    repr: KeyRepr,
    size: u64,
}

impl MemoKey {
    /// Converts a value to a key. Returns `None` for a value that is not a valid key.
    pub fn from_value(v: &Value) -> Option<MemoKey> {
        MemoKey::build(v, MemoKey::size_of(v)?)
    }

    /// Converts a value of key size `size`, as measured by [`MemoKey::size_of`],
    /// to a key. The work is linear in the number of distinct pairs of `v`.
    /// Returns `None` for a value that is not a valid key.
    pub(crate) fn build(v: &Value, size: u64) -> Option<MemoKey> {
        let repr = if size <= FLAT_KEY_MAX {
            KeyRepr::Flat(MemoKey::flat(v)?)
        } else {
            KeyRepr::Dag(MemoKey::dag(v)?)
        };
        #[cfg(test)]
        KEY_ATOMS_BUILT.with(|n| {
            let built = match &repr {
                KeyRepr::Flat(atoms) => atoms.len(),
                KeyRepr::Dag(nodes) => nodes.len(),
            };
            n.set(n.get() + built as u64)
        });
        Some(MemoKey { repr, size })
    }

    /// The pre-order atoms of `v`, or `None` for an invalid key. Call it only on a
    /// value whose size is known to be small.
    fn flat(v: &Value) -> Option<Vec<KeyAtom>> {
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
        Some(atoms)
    }

    /// The distinct subvalues of `v` in canonical order (see [`KeyRepr::Dag`]), or
    /// `None` for an invalid key. Each pair in memory is visited once.
    fn dag(v: &Value) -> Option<Vec<KeyNode>> {
        let mut nodes: Vec<KeyNode> = Vec::new();
        let mut ids: FxHashMap<KeyNode, usize> = FxHashMap::default();
        let mut by_ptr: FxHashMap<*const (Value, Value), usize> = FxHashMap::default();
        let mut intern = |node: KeyNode, nodes: &mut Vec<KeyNode>| -> usize {
            *ids.entry(node).or_insert_with(|| {
                nodes.push(node);
                nodes.len() - 1
            })
        };
        // `(value, components done)`; `results` holds the finished node ids.
        let mut work: Vec<(&Value, bool)> = vec![(v, false)];
        let mut results: Vec<usize> = Vec::new();
        while let Some((v, done)) = work.pop() {
            let id = match v {
                Value::Int(n) => intern(KeyNode::Int(*n), &mut nodes),
                Value::Bool(b) => intern(KeyNode::Bool(*b), &mut nodes),
                Value::Unit => intern(KeyNode::Unit, &mut nodes),
                Value::Pair(p) => {
                    let ptr = Rc::as_ptr(p);
                    if done {
                        let b = results.pop()?;
                        let a = results.pop()?;
                        let id = intern(KeyNode::Pair(a, b), &mut nodes);
                        by_ptr.insert(ptr, id);
                        id
                    } else if let Some(&id) = by_ptr.get(&ptr) {
                        id
                    } else {
                        work.push((v, true));
                        work.push((&p.1, false));
                        work.push((&p.0, false));
                        continue;
                    }
                }
                _ => return None,
            };
            results.push(id);
        }
        Some(nodes)
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

    /// The key size in value nodes, as the cost model counts it.
    pub fn size(&self) -> u64 {
        self.size
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
///
/// Only [`List::cons`] builds a cell, so the cached length is always right: the
/// cost of `length`, `array_of_list` and `eq` depends on it.
///
/// ```compile_fail,E0451
/// use dc_eval::value::{Cons, List};
/// use dc_eval::Value;
/// use std::rc::Rc;
/// let wrong = List(Some(Rc::new(Cons { head: Value::Unit, tail: List::nil(), len: 5 })));
/// ```
pub struct Cons {
    /// The first element.
    pub head: Value,
    /// The rest of the list.
    pub tail: List,
    /// The length of this list (at least 1). Private, so no cell outside this
    /// module can carry a wrong one.
    len: u64,
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

#[cfg(test)]
thread_local! {
    /// Key atoms or distinct key nodes built on this thread, so tests can check
    /// how much of a key is built.
    pub(crate) static KEY_ATOMS_BUILT: Cell<u64> = const { Cell::new(0) };
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

/// A pair of shared value parts that [`structural_eq`] has already compared: a
/// kind tag (value, list cell or tree node, whose addresses can coincide) and
/// the two addresses.
pub(crate) type SeenPairs = FxHashSet<(u8, *const (), *const ())>;

/// The identity of a compound pair of parts, if it can be shared: pairs,
/// options, arrays, list cells and tree nodes. Scalars, `nil` and `leaf` are
/// cheaper to compare again than to remember.
fn pair_id(item: &EqWork<'_>) -> Option<(u8, *const (), *const ())> {
    match item {
        EqWork::Value(x @ (Value::Pair(_) | Value::Option(Some(_)) | Value::Array(_)), y) => {
            Some((0, ptr::from_ref(*x).cast(), ptr::from_ref(*y).cast()))
        }
        EqWork::List(List(Some(c)), List(Some(d))) => {
            Some((1, Rc::as_ptr(c).cast(), Rc::as_ptr(d).cast()))
        }
        EqWork::Tree(Tree(Some(m)), Tree(Some(n))) => {
            Some((2, Rc::as_ptr(m).cast(), Rc::as_ptr(n).cast()))
        }
        _ => None,
    }
}

/// A pair of parts still to compare in [`structural_eq`].
enum EqWork<'a> {
    Value(&'a Value, &'a Value),
    List(&'a List, &'a List),
    Tree(&'a Tree, &'a Tree),
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
///
/// With `seen`, a compound pair of parts at the same two addresses as one already
/// compared is skipped and not counted: values are immutable, so it compares the
/// same way again, and the first comparison's result already decides the answer.
/// The data result is unchanged, but the count is then of distinct pairs, not the
/// cost model's count, so the machine passes `None`.
pub(crate) fn structural_eq(
    a: &Value,
    b: &Value,
    limit: u64,
    mut seen: Option<&mut SeenPairs>,
) -> (EqResult, u64) {
    use EqWork as Work;
    let mut work = vec![Work::Value(a, b)];
    let mut nodes = 0u64;
    while let Some(item) = work.pop() {
        if let (Some(seen), Some(id)) = (seen.as_deref_mut(), pair_id(&item)) {
            if !seen.insert(id) {
                continue;
            }
        }
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

/// The number of values and tree nodes `Debug` writes before it elides the rest
/// with `...`.
pub const DEBUG_NODES: u64 = 1_000;

/// The marker for elided parts in bounded output.
const ELIDED: &str = "...";

/// Writes at most [`DEBUG_NODES`] values and tree nodes, in the same syntax as
/// `Display`, so that test failures and logged errors stay short however large a
/// value is.
impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_value(Piece::Value(self), DEBUG_NODES, f)
    }
}

/// Bounded like `Debug` for [`Value`], with the list syntax `[1; 2]`.
impl fmt::Debug for List {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_value(Piece::List(self), DEBUG_NODES, f)
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

/// Bounded like `Debug` for [`Value`].
impl fmt::Debug for Tree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_value(Piece::Tree(self), DEBUG_NODES, f)
    }
}

/// A piece of output still to write.
enum Piece<'a> {
    Value(&'a Value),
    List(&'a List),
    Tree(&'a Tree),
    Text(&'static str),
    /// The elements of a list after its first, each preceded by `; `.
    ListRest(&'a List),
    /// The elements of an array from the index on, each preceded by `; `.
    ArrayRest(&'a imbl::Vector<Value>, usize),
}

/// Writes a value with an explicit work stack, so deeply nested values format
/// without host recursion.
///
/// At most `limit` values and tree nodes are written. After that, each part not
/// yet written is written as `...` (the rest of a list or an array as one `...`),
/// and the brackets still open are closed. The work stack grows by a constant per
/// node written, so the output and the work are both O(`limit`), whatever the
/// size of the value.
fn write_value(start: Piece<'_>, limit: u64, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let mut work = vec![start];
    let mut written = 0u64;
    while let Some(piece) = work.pop() {
        let full = written >= limit;
        match piece {
            Piece::Text(s) => f.write_str(s)?,
            Piece::ListRest(l) => {
                if let Some(c) = &l.0 {
                    if full {
                        write!(f, "; {ELIDED}")?;
                    } else {
                        work.push(Piece::ListRest(&c.tail));
                        work.push(Piece::Value(&c.head));
                        work.push(Piece::Text("; "));
                    }
                }
            }
            Piece::ArrayRest(a, i) => {
                if let Some(x) = a.get(i) {
                    if full {
                        write!(f, "; {ELIDED}")?;
                    } else {
                        work.push(Piece::ArrayRest(a, i + 1));
                        work.push(Piece::Value(x));
                        work.push(Piece::Text("; "));
                    }
                }
            }
            Piece::List(l) => match &l.0 {
                None => f.write_str("[]")?,
                Some(c) => {
                    work.push(Piece::Text("]"));
                    work.push(Piece::ListRest(&c.tail));
                    work.push(Piece::Value(&c.head));
                    work.push(Piece::Text("["));
                }
            },
            // A tree value is written as its root node, which is counted below.
            Piece::Value(Value::Tree(t)) => work.push(Piece::Tree(t)),
            Piece::Value(_) | Piece::Tree(_) if full => f.write_str(ELIDED)?,
            Piece::Tree(t) => {
                written += 1;
                match &t.0 {
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
                }
            }
            Piece::Value(v) => {
                written += 1;
                match v {
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
                    Value::List(l) => work.push(Piece::List(l)),
                    Value::Array(a) => match a.front() {
                        None => f.write_str("[||]")?,
                        Some(x) => {
                            work.push(Piece::Text("|]"));
                            work.push(Piece::ArrayRest(a, 1));
                            work.push(Piece::Value(x));
                            work.push(Piece::Text("[|"));
                        }
                    },
                    Value::Tree(t) => work.push(Piece::Tree(t)),
                    Value::Fun(_) => f.write_str("<fun>")?,
                }
            }
        }
    }
    Ok(())
}

/// Writes the whole value as a tree. A value with shared parts can be
/// exponentially longer written out than it is in memory, so use this only on
/// values known to be small; [`Value::display_within`] bounds the output.
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_value(Piece::Value(self), u64::MAX, f)
    }
}

impl Value {
    /// Formats at most `limit` values and tree nodes of this value, in the
    /// `Display` syntax, and each part beyond them as `...`. The output and the
    /// work are O(`limit`), however large the value is as a tree. With a limit
    /// at least the value's size the output is the same as `Display`'s.
    pub fn display_within(&self, limit: u64) -> DisplayWithin<'_> {
        DisplayWithin { value: self, limit }
    }
}

/// A value formatted with a bound; see [`Value::display_within`].
#[derive(Clone, Copy, Debug)]
pub struct DisplayWithin<'a> {
    value: &'a Value,
    limit: u64,
}

impl fmt::Display for DisplayWithin<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_value(Piece::Value(self.value), self.limit, f)
    }
}

/// Values above this many node pairs are compared with [`SeenPairs`].
const PLAIN_EQ_NODES: u64 = 1 << 12;

/// Structural equality for tests, tools and task checks. Function values are
/// never equal, even to themselves.
///
/// Exact, and bounded by the memory the values use rather than their size as
/// trees. Values up to 4096 node pairs are compared directly. Larger ones are
/// compared again, remembering the pairs of parts already compared, so each
/// distinct pair of parts (by address) is compared once. That is at most the
/// product of the two values' part counts in memory, and linear in them when
/// the values have the same sharing, as two runs of one program do.
impl PartialEq for Value {
    fn eq(&self, other: &Value) -> bool {
        match structural_eq(self, other, PLAIN_EQ_NODES, None).0 {
            EqResult::Data(same) => same,
            EqResult::Function => false,
            EqResult::OverLimit => {
                let mut seen = SeenPairs::default();
                let (result, _) = structural_eq(self, other, u64::MAX, Some(&mut seen));
                result == EqResult::Data(true)
            }
        }
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
