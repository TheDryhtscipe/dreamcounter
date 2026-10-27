//! The CEK machine (spec §4, decision D6).
//!
//! The control is either a term to evaluate in an environment or a value to return
//! to the topmost continuation frame. Continuations live in an explicit `Vec`, so a
//! dc program can recurse as deep as memory allows without touching the Rust stack.
//! Every step is charged through [`crate::cost`], in one place ([`Machine::charge`]),
//! which is also where the global fuel is checked.

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use dc_core::{Arena, LibId, Prim, Term, TermId};

use crate::cost;
use crate::value::{structural_eq, Env, EqResult, Fun, List, MemoKey, MemoTable, Tree, Value};

/// The result of one evaluation (spec §4). The `u64` is the number of steps taken.
#[derive(Clone, Debug)]
pub enum Outcome {
    /// Evaluation finished with a value.
    Value(Value, u64),
    /// Evaluation stopped with a runtime error.
    Error(ErrorKind, u64),
    /// The global fuel ran out. The count is the first one above the fuel.
    Timeout(u64),
}

impl Outcome {
    /// The number of steps taken.
    pub fn steps(&self) -> u64 {
        match self {
            Outcome::Value(_, s) | Outcome::Error(_, s) | Outcome::Timeout(s) => *s,
        }
    }

    /// The value, if evaluation finished with one.
    pub fn value(&self) -> Option<&Value> {
        match self {
            Outcome::Value(v, _) => Some(v),
            _ => None,
        }
    }
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Outcome::Value(v, s) => write!(f, "Value({v}, {s})"),
            Outcome::Error(k, s) => write!(f, "Error({k:?}, {s})"),
            Outcome::Timeout(s) => write!(f, "Timeout({s})"),
        }
    }
}

/// A runtime error (spec §4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// `div` or `mod` by zero.
    DivisionByZero,
    /// Checked 64-bit arithmetic overflowed, including `div`/`mod` of `i64::MIN`
    /// by -1.
    Overflow,
    /// `array_get` or `array_set` with an index outside the array.
    IndexOutOfBounds,
    /// A `bfix` call with no depth fuel left, or a `memorec` miss with no misses
    /// left (spec §6).
    FuelExhausted,
    /// `eq` reached a function value.
    EqOnFunction,
    /// A `memorec` key that is not built from `Int`, `Bool`, `Unit` and pairs.
    InvalidMemoKey,
    /// A `memorec` `self` was called after its outermost `memorec` call returned
    /// and dropped the memo table. Charged 0 steps.
    MemoSelfEscaped,
    /// The machine cannot continue. Only reachable for ill-typed or unexpanded terms.
    Stuck(Stuck),
}

/// Why the machine is stuck. A well-typed, fully expanded closed term never gets
/// stuck.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stuck {
    /// A de Bruijn index with no binder: the term was not closed.
    UnboundVar(u32),
    /// A non-function value was applied.
    NotAFunction,
    /// A primitive received an argument of the wrong kind, or a function argument
    /// of a combinator returned one.
    BadArgument(Prim),
    /// An `if`, `and` or `or` decided on a non-boolean.
    NotABool,
    /// A library reference reached the evaluator: abstractions must be expanded
    /// before evaluation (spec §8).
    UnexpandedLib(LibId),
}

/// No fuel limit in practice.
pub const UNLIMITED: u64 = u64::MAX;

/// Evaluates the closed term `term` with a global budget of `fuel` steps.
pub fn eval(arena: &Arena, term: TermId, fuel: u64) -> Outcome {
    Machine::new(arena, fuel).run(Control::Eval(term, Env::empty()))
}

/// Evaluates the closed term `program`, then applies the result to each of
/// `inputs` in turn, all within one budget of `fuel` steps. Applying a closure to
/// an input is a β step like any other.
pub fn apply(arena: &Arena, program: TermId, inputs: &[Value], fuel: u64) -> Outcome {
    let mut m = Machine::new(arena, fuel);
    for x in inputs.iter().rev() {
        m.stack.push(Frame::ApplyTo(x.clone()));
    }
    m.run(Control::Eval(program, Env::empty()))
}

enum Control {
    Eval(TermId, Env),
    Return(Value),
}

enum Halt {
    Error(ErrorKind),
    Timeout,
}

type Step = Result<Control, Halt>;

fn stuck(s: Stuck) -> Halt {
    Halt::Error(ErrorKind::Stuck(s))
}

fn bad(p: Prim) -> Halt {
    stuck(Stuck::BadArgument(p))
}

fn error(k: ErrorKind) -> Halt {
    Halt::Error(k)
}

/// A `tree_fold` work item.
enum TreeTask {
    /// Fold this subtree and push its result.
    Visit(Tree),
    /// Pop the left and right results and call `f left value right`.
    Combine(Value),
}

/// A continuation frame. Each waits for one value.
enum Frame {
    /// The function of an application is done: evaluate the argument next.
    AppArg { arg: TermId, env: Env },
    /// The argument is done: apply this function to it.
    AppFun { f: Value },
    /// A function value is done: apply it to this argument.
    ApplyTo(Value),
    /// The bound value of a `let` is done: evaluate the body.
    LetBody { body: TermId, env: Env },
    /// The condition is done.
    If {
        then_: TermId,
        else_: TermId,
        env: Env,
    },
    /// The left operand of `and` is done.
    And { rhs: TermId, env: Env },
    /// The left operand of `or` is done.
    Or { rhs: TermId, env: Env },
    /// `foldl`: the new accumulator is done; fold over `rest` next.
    Foldl { f: Value, rest: List },
    /// `foldr`: the new accumulator is done; `elems` are the elements still to
    /// fold, the next one last.
    Foldr { f: Value, elems: Vec<Value> },
    /// `loop`: the accumulator after iteration `i - 1` is done.
    Loop { f: Value, i: i64, n: i64 },
    /// `unfold`: the generator's result is done; `out` holds the elements so far.
    Unfold { g: Value, out: Vec<Value> },
    /// `array_tabulate`: element `i` is done.
    Tabulate {
        f: Value,
        i: i64,
        n: i64,
        out: imbl::Vector<Value>,
    },
    /// `tree_fold`: the result of one `f` call is done.
    TreeFold {
        z: Value,
        f: Value,
        tasks: Vec<TreeTask>,
        results: Vec<Value>,
    },
    /// A `memorec` miss is done: store the result under `key`.
    MemoStore {
        table: Rc<RefCell<MemoTable>>,
        key: MemoKey,
    },
    /// The outermost call of one `memorec` is done. This frame owns the memo table
    /// and drops it on return (spec §6); every `self` holds only a weak reference.
    MemoScope { table: Rc<RefCell<MemoTable>> },
}

struct Machine<'a> {
    arena: &'a Arena,
    stack: Vec<Frame>,
    steps: u64,
    fuel: u64,
}

impl<'a> Machine<'a> {
    fn new(arena: &'a Arena, fuel: u64) -> Machine<'a> {
        Machine {
            arena,
            stack: Vec::new(),
            steps: 0,
            fuel,
        }
    }

    fn run(&mut self, start: Control) -> Outcome {
        let mut control = start;
        loop {
            let next = match control {
                Control::Eval(t, env) => self.eval_term(t, env),
                Control::Return(v) => match self.stack.pop() {
                    None => return Outcome::Value(v, self.steps),
                    Some(frame) => self.resume(frame, v),
                },
            };
            control = match next {
                Ok(c) => c,
                Err(Halt::Error(k)) => return Outcome::Error(k, self.steps),
                Err(Halt::Timeout) => return Outcome::Timeout(self.steps),
            };
        }
    }

    /// Charges `n` steps. Stops with a timeout at the first charge that takes the
    /// count above the fuel.
    fn charge(&mut self, n: u64) -> Result<(), Halt> {
        self.steps = self.steps.saturating_add(n);
        if self.steps > self.fuel {
            Err(Halt::Timeout)
        } else {
            Ok(())
        }
    }

    fn eval_term(&mut self, t: TermId, env: Env) -> Step {
        Ok(match self.arena.get(t) {
            Term::Var(i) => {
                Control::Return(env.lookup(i).cloned().ok_or(stuck(Stuck::UnboundVar(i)))?)
            }
            Term::Lam(body) => Control::Return(Value::Fun(Rc::new(Fun::Closure { body, env }))),
            Term::App(f, a) => {
                self.stack.push(Frame::AppArg {
                    arg: a,
                    env: env.clone(),
                });
                Control::Eval(f, env)
            }
            Term::Let(x, body) => {
                self.stack.push(Frame::LetBody {
                    body,
                    env: env.clone(),
                });
                Control::Eval(x, env)
            }
            Term::Int(n) => Control::Return(Value::Int(n)),
            Term::Bool(b) => Control::Return(Value::Bool(b)),
            Term::Unit => Control::Return(Value::Unit),
            Term::Prim(p) => match p {
                Prim::Nil => {
                    self.charge(cost::PRIM)?;
                    Control::Return(Value::List(List::nil()))
                }
                Prim::None => {
                    self.charge(cost::PRIM)?;
                    Control::Return(Value::none())
                }
                Prim::Leaf => {
                    self.charge(cost::PRIM)?;
                    Control::Return(Value::Tree(Tree::leaf()))
                }
                _ => Control::Return(Value::Fun(Rc::new(Fun::Partial {
                    prim: p,
                    args: Vec::new(),
                }))),
            },
            Term::Lib(l) => return Err(stuck(Stuck::UnexpandedLib(l))),
            Term::If(c, then_, else_) => {
                self.stack.push(Frame::If {
                    then_,
                    else_,
                    env: env.clone(),
                });
                Control::Eval(c, env)
            }
            Term::And(a, b) => {
                self.stack.push(Frame::And {
                    rhs: b,
                    env: env.clone(),
                });
                Control::Eval(a, env)
            }
            Term::Or(a, b) => {
                self.stack.push(Frame::Or {
                    rhs: b,
                    env: env.clone(),
                });
                Control::Eval(a, env)
            }
        })
    }

    fn decide(&mut self, v: &Value) -> Result<bool, Halt> {
        match v {
            Value::Bool(b) => {
                self.charge(cost::DECISION)?;
                Ok(*b)
            }
            _ => Err(stuck(Stuck::NotABool)),
        }
    }

    fn resume(&mut self, frame: Frame, v: Value) -> Step {
        match frame {
            Frame::AppArg { arg, env } => {
                self.stack.push(Frame::AppFun { f: v });
                Ok(Control::Eval(arg, env))
            }
            Frame::AppFun { f } => self.apply(&f, v),
            Frame::ApplyTo(x) => self.apply(&v, x),
            Frame::LetBody { body, env } => Ok(Control::Eval(body, env.push(v))),
            Frame::If { then_, else_, env } => {
                let b = self.decide(&v)?;
                Ok(Control::Eval(if b { then_ } else { else_ }, env))
            }
            Frame::And { rhs, env } => Ok(if self.decide(&v)? {
                Control::Eval(rhs, env)
            } else {
                Control::Return(Value::Bool(false))
            }),
            Frame::Or { rhs, env } => Ok(if self.decide(&v)? {
                Control::Return(Value::Bool(true))
            } else {
                Control::Eval(rhs, env)
            }),
            Frame::Foldl { f, rest } => self.foldl_next(f, rest, v),
            Frame::Foldr { f, elems } => self.foldr_next(f, elems, v),
            Frame::Loop { f, i, n } => self.loop_next(f, i, n, v),
            Frame::Unfold { g, mut out } => match &v {
                Value::Option(None) => Ok(Control::Return(Value::list(out))),
                Value::Option(Some(p)) => match &**p {
                    Value::Pair(p) => {
                        self.charge(cost::ITERATION)?;
                        out.push(p.0.clone());
                        let next = p.1.clone();
                        self.stack.push(Frame::Unfold { g: g.clone(), out });
                        self.apply(&g, next)
                    }
                    _ => Err(bad(Prim::Unfold)),
                },
                _ => Err(bad(Prim::Unfold)),
            },
            Frame::Tabulate { f, i, n, mut out } => {
                out.push_back(v);
                self.tabulate_next(f, i + 1, n, out)
            }
            Frame::TreeFold {
                z,
                f,
                tasks,
                mut results,
            } => {
                results.push(v);
                self.tree_fold_run(z, f, tasks, results)
            }
            Frame::MemoStore { table, key } => {
                table.borrow_mut().entries.insert(key, v.clone());
                Ok(Control::Return(v))
            }
            Frame::MemoScope { table } => {
                drop(table);
                Ok(Control::Return(v))
            }
        }
    }

    /// Applies the function value `f` to `x`.
    fn apply(&mut self, f: &Value, x: Value) -> Step {
        let Value::Fun(fun) = f else {
            return Err(stuck(Stuck::NotAFunction));
        };
        match &**fun {
            Fun::Closure { body, env } => {
                self.charge(cost::BETA)?;
                Ok(Control::Eval(*body, env.push(x)))
            }
            Fun::Partial { prim, args } => {
                let mut args = args.clone();
                args.push(x);
                if args.len() < prim.arity() {
                    self.charge(cost::PARTIAL)?;
                    Ok(Control::Return(Value::Fun(Rc::new(Fun::Partial {
                        prim: *prim,
                        args,
                    }))))
                } else {
                    self.prim(*prim, &args)
                }
            }
            Fun::BfixSelf { fuel, f } => self.bfix_call(*fuel, f, x),
            Fun::MemoSelf { table, f } => match table.upgrade() {
                Some(table) => self.memo_call(&table, f, x),
                None => Err(error(ErrorKind::MemoSelfEscaped)),
            },
        }
    }

    /// Runs a saturated primitive.
    fn prim(&mut self, p: Prim, args: &[Value]) -> Step {
        use Value as V;
        let ret = |v: Value| Ok(Control::Return(v));
        match (p, args) {
            (Prim::Add, [V::Int(a), V::Int(b)]) => self.arith(a.checked_add(*b)),
            (Prim::Sub, [V::Int(a), V::Int(b)]) => self.arith(a.checked_sub(*b)),
            (Prim::Mul, [V::Int(a), V::Int(b)]) => self.arith(a.checked_mul(*b)),
            (Prim::Div, [V::Int(a), V::Int(b)]) => {
                self.charge(cost::PRIM)?;
                if *b == 0 {
                    return Err(error(ErrorKind::DivisionByZero));
                }
                ret(V::Int(a.checked_div(*b).ok_or(error(ErrorKind::Overflow))?))
            }
            (Prim::Mod, [V::Int(a), V::Int(b)]) => {
                self.charge(cost::PRIM)?;
                if *b == 0 {
                    return Err(error(ErrorKind::DivisionByZero));
                }
                ret(V::Int(a.checked_rem(*b).ok_or(error(ErrorKind::Overflow))?))
            }
            (Prim::IntEq, [V::Int(a), V::Int(b)]) => self.constant(V::Bool(a == b)),
            (Prim::Lt, [V::Int(a), V::Int(b)]) => self.constant(V::Bool(a < b)),
            (Prim::Le, [V::Int(a), V::Int(b)]) => self.constant(V::Bool(a <= b)),
            (Prim::Not, [V::Bool(b)]) => self.constant(V::Bool(!b)),
            (Prim::Eq, [a, b]) => {
                let (result, nodes) = structural_eq(a, b);
                self.charge(cost::eq(nodes))?;
                match result {
                    EqResult::Data(same) => ret(V::Bool(same)),
                    EqResult::Function => Err(error(ErrorKind::EqOnFunction)),
                }
            }
            (Prim::Pair, [a, b]) => self.constant(V::pair(a.clone(), b.clone())),
            (Prim::Fst, [V::Pair(p)]) => self.constant(p.0.clone()),
            (Prim::Snd, [V::Pair(p)]) => self.constant(p.1.clone()),
            (Prim::Some, [a]) => self.constant(V::some(a.clone())),
            (Prim::Cons, [x, V::List(xs)]) => {
                self.constant(V::List(List::cons(x.clone(), xs.clone())))
            }
            (Prim::Length, [V::List(xs)]) => self.constant(V::Int(len_to_int(xs.len()))),
            (Prim::Node, [V::Tree(l), x, V::Tree(r)]) => {
                self.constant(V::Tree(Tree::node(l.clone(), x.clone(), r.clone())))
            }
            (Prim::OptionCase, [V::Option(o), on_none, on_some]) => {
                self.charge(cost::PRIM)?;
                match o {
                    None => self.apply(on_none, V::Unit),
                    Some(x) => self.apply(on_some, (**x).clone()),
                }
            }
            (Prim::ListCase, [V::List(xs), on_nil, on_cons]) => {
                self.charge(cost::PRIM)?;
                match &xs.0 {
                    None => self.apply(on_nil, V::Unit),
                    Some(cell) => {
                        self.stack.push(Frame::ApplyTo(V::List(cell.tail.clone())));
                        self.apply(on_cons, cell.head.clone())
                    }
                }
            }
            (Prim::TreeCase, [V::Tree(t), on_leaf, on_node]) => {
                self.charge(cost::PRIM)?;
                match &t.0 {
                    None => self.apply(on_leaf, V::Unit),
                    Some(node) => {
                        self.stack.push(Frame::ApplyTo(V::Tree(node.right.clone())));
                        self.stack.push(Frame::ApplyTo(node.value.clone()));
                        self.apply(on_node, V::Tree(node.left.clone()))
                    }
                }
            }
            (Prim::ArrayTabulate, [V::Int(n), f]) => {
                self.tabulate_next(f.clone(), 0, *n, imbl::Vector::new())
            }
            (Prim::ArrayGet, [V::Array(a), V::Int(i)]) => {
                self.charge(cost::PRIM)?;
                let x = index(a, *i).and_then(|i| a.get(i)).cloned();
                ret(x.ok_or(error(ErrorKind::IndexOutOfBounds))?)
            }
            (Prim::ArraySet, [V::Array(a), V::Int(i), x]) => {
                self.charge(cost::array_set(a.len() as u64))?;
                let i = index(a, *i).ok_or(error(ErrorKind::IndexOutOfBounds))?;
                ret(V::Array(a.update(i, x.clone())))
            }
            (Prim::ArraySize, [V::Array(a)]) => self.constant(V::Int(len_to_int(a.len() as u64))),
            (Prim::ArrayOfList, [V::List(xs)]) => {
                self.charge(cost::array_convert(xs.len()))?;
                ret(V::Array(xs.iter().cloned().collect()))
            }
            (Prim::ArrayToList, [V::Array(a)]) => {
                self.charge(cost::array_convert(a.len() as u64))?;
                let items: Vec<Value> = a.iter().cloned().collect();
                ret(V::list(items))
            }
            (Prim::TreeFold, [z, f, V::Tree(t)]) => self.tree_fold_run(
                z.clone(),
                f.clone(),
                vec![TreeTask::Visit(t.clone())],
                Vec::new(),
            ),
            (Prim::Foldl, [f, z, V::List(xs)]) => self.foldl_next(f.clone(), xs.clone(), z.clone()),
            (Prim::Foldr, [f, z, V::List(xs)]) => {
                let elems: Vec<Value> = xs.iter().cloned().collect();
                self.foldr_next(f.clone(), elems, z.clone())
            }
            (Prim::Loop, [V::Int(n), z, f]) => self.loop_next(f.clone(), 0, *n, z.clone()),
            (Prim::Unfold, [g, s]) => {
                self.stack.push(Frame::Unfold {
                    g: g.clone(),
                    out: Vec::new(),
                });
                self.apply(g, s.clone())
            }
            (Prim::Bfix, [V::Int(n), f, x]) => self.bfix_call(*n, f, x.clone()),
            (Prim::Memorec, [V::Int(n), f, x]) => {
                let table = Rc::new(RefCell::new(MemoTable {
                    entries: Default::default(),
                    misses_left: *n,
                }));
                self.stack.push(Frame::MemoScope {
                    table: table.clone(),
                });
                self.memo_call(&table, f, x.clone())
            }
            _ => Err(bad(p)),
        }
    }

    /// A constant-cost primitive returning `v`.
    fn constant(&mut self, v: Value) -> Step {
        self.charge(cost::PRIM)?;
        Ok(Control::Return(v))
    }

    /// A checked arithmetic primitive.
    fn arith(&mut self, r: Option<i64>) -> Step {
        self.charge(cost::PRIM)?;
        Ok(Control::Return(Value::Int(
            r.ok_or(error(ErrorKind::Overflow))?,
        )))
    }

    fn foldl_next(&mut self, f: Value, rest: List, acc: Value) -> Step {
        match &rest.0 {
            None => Ok(Control::Return(acc)),
            Some(cell) => {
                self.charge(cost::ITERATION)?;
                self.stack.push(Frame::Foldl {
                    f: f.clone(),
                    rest: cell.tail.clone(),
                });
                self.stack.push(Frame::ApplyTo(cell.head.clone()));
                self.apply(&f, acc)
            }
        }
    }

    fn foldr_next(&mut self, f: Value, mut elems: Vec<Value>, acc: Value) -> Step {
        match elems.pop() {
            None => Ok(Control::Return(acc)),
            Some(x) => {
                self.charge(cost::ITERATION)?;
                self.stack.push(Frame::Foldr {
                    f: f.clone(),
                    elems,
                });
                self.stack.push(Frame::ApplyTo(acc));
                self.apply(&f, x)
            }
        }
    }

    fn loop_next(&mut self, f: Value, i: i64, n: i64, acc: Value) -> Step {
        if i >= n {
            return Ok(Control::Return(acc));
        }
        self.charge(cost::ITERATION)?;
        self.stack.push(Frame::Loop {
            f: f.clone(),
            i: i + 1,
            n,
        });
        self.stack.push(Frame::ApplyTo(acc));
        self.apply(&f, Value::Int(i))
    }

    fn tabulate_next(&mut self, f: Value, i: i64, n: i64, out: imbl::Vector<Value>) -> Step {
        if i >= n {
            return Ok(Control::Return(Value::Array(out)));
        }
        self.charge(cost::ITERATION)?;
        self.stack.push(Frame::Tabulate {
            f: f.clone(),
            i,
            n,
            out,
        });
        self.apply(&f, Value::Int(i))
    }

    /// Folds a tree in post-order: left subtree, right subtree, then
    /// `f left value right`, one `f` call at a time through the machine.
    fn tree_fold_run(
        &mut self,
        z: Value,
        f: Value,
        mut tasks: Vec<TreeTask>,
        mut results: Vec<Value>,
    ) -> Step {
        while let Some(task) = tasks.pop() {
            match task {
                TreeTask::Visit(t) => match &t.0 {
                    None => results.push(z.clone()),
                    Some(node) => {
                        tasks.push(TreeTask::Combine(node.value.clone()));
                        tasks.push(TreeTask::Visit(node.right.clone()));
                        tasks.push(TreeTask::Visit(node.left.clone()));
                    }
                },
                TreeTask::Combine(x) => {
                    let (Some(right), Some(left)) = (results.pop(), results.pop()) else {
                        return Err(bad(Prim::TreeFold));
                    };
                    self.charge(cost::ITERATION)?;
                    let g = f.clone();
                    self.stack.push(Frame::TreeFold {
                        z,
                        f,
                        tasks,
                        results,
                    });
                    self.stack.push(Frame::ApplyTo(right));
                    self.stack.push(Frame::ApplyTo(x));
                    return self.apply(&g, left);
                }
            }
        }
        match results.pop() {
            Some(v) => Ok(Control::Return(v)),
            None => Err(bad(Prim::TreeFold)),
        }
    }

    /// `bfix fuel f x`: one call of a bounded fixpoint.
    fn bfix_call(&mut self, fuel: i64, f: &Value, x: Value) -> Step {
        if fuel <= 0 {
            return Err(error(ErrorKind::FuelExhausted));
        }
        self.charge(cost::ITERATION)?;
        let this = Value::Fun(Rc::new(Fun::BfixSelf {
            fuel: fuel - 1,
            f: f.clone(),
        }));
        self.stack.push(Frame::ApplyTo(x));
        self.apply(f, this)
    }

    /// One memo call on `x`: a hit returns the stored result, a miss evaluates
    /// `f self x` and stores the result.
    fn memo_call(&mut self, table: &Rc<RefCell<MemoTable>>, f: &Value, x: Value) -> Step {
        let key = MemoKey::from_value(&x).ok_or(error(ErrorKind::InvalidMemoKey))?;
        let size = key.size();
        let hit = table.borrow().entries.get(&key).cloned();
        if let Some(v) = hit {
            self.charge(cost::memo_call(size))?;
            return Ok(Control::Return(v));
        }
        {
            let mut t = table.borrow_mut();
            if t.misses_left <= 0 {
                return Err(error(ErrorKind::FuelExhausted));
            }
            t.misses_left -= 1;
        }
        self.charge(cost::memo_call(size))?;
        self.stack.push(Frame::MemoStore {
            table: table.clone(),
            key,
        });
        self.stack.push(Frame::ApplyTo(x));
        let this = Value::Fun(Rc::new(Fun::MemoSelf {
            table: Rc::downgrade(table),
            f: f.clone(),
        }));
        self.apply(f, this)
    }
}

/// An array index as `usize`, if it is in bounds.
fn index(a: &imbl::Vector<Value>, i: i64) -> Option<usize> {
    usize::try_from(i).ok().filter(|&i| i < a.len())
}

fn len_to_int(n: u64) -> i64 {
    // A list or array of 2⁶³ elements cannot exist in memory.
    i64::try_from(n).unwrap_or(i64::MAX)
}
