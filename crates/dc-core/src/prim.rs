//! Primitives and recursion combinators (spec §5, §6).
//!
//! Every primitive has a canonical name, used by the core syntax and by Stitch
//! serialisation, and a fixed arity. Nullary primitives (`nil`, `none`, `leaf`) are
//! constants. Surface operators such as `+` and `::` map onto these names in
//! `dc-surface`.

use std::fmt;

macro_rules! prims {
    ($($variant:ident => $name:literal, $arity:literal;)*) => {
        /// A built-in primitive or combinator.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum Prim {
            $($variant,)*
        }

        impl Prim {
            /// Every primitive, in declaration order.
            pub const ALL: &'static [Prim] = &[$(Prim::$variant,)*];

            /// The canonical name, as written in core syntax.
            pub fn name(self) -> &'static str {
                match self {
                    $(Prim::$variant => $name,)*
                }
            }

            /// The number of arguments the primitive takes before it runs.
            pub fn arity(self) -> usize {
                match self {
                    $(Prim::$variant => $arity,)*
                }
            }

            /// Looks a primitive up by its canonical name.
            pub fn from_name(name: &str) -> Option<Prim> {
                match name {
                    $($name => Some(Prim::$variant),)*
                    _ => None,
                }
            }
        }
    };
}

prims! {
    // Integers (checked arithmetic).
    Add => "add", 2;
    Sub => "sub", 2;
    Mul => "mul", 2;
    Div => "div", 2;
    Mod => "mod", 2;
    IntEq => "int_eq", 2;
    Lt => "lt", 2;
    Le => "le", 2;
    // Booleans.
    Not => "not", 1;
    // Structural equality on non-function values.
    Eq => "eq", 2;
    // Pairs.
    Pair => "pair", 2;
    Fst => "fst", 1;
    Snd => "snd", 1;
    // Options.
    None => "none", 0;
    Some => "some", 1;
    OptionCase => "option_case", 3;
    // Lists.
    Nil => "nil", 0;
    Cons => "cons", 2;
    ListCase => "list_case", 3;
    Length => "length", 1;
    // Persistent arrays.
    ArrayTabulate => "array_tabulate", 2;
    ArrayGet => "array_get", 2;
    ArraySet => "array_set", 3;
    ArraySize => "array_size", 1;
    ArrayOfList => "array_of_list", 1;
    ArrayToList => "array_to_list", 1;
    // Binary trees.
    Leaf => "leaf", 0;
    Node => "node", 3;
    TreeCase => "tree_case", 3;
    TreeFold => "tree_fold", 3;
    // Recursion combinators (spec §6).
    Foldl => "foldl", 3;
    Foldr => "foldr", 3;
    Loop => "loop", 3;
    Unfold => "unfold", 2;
    Bfix => "bfix", 3;
    Memorec => "memorec", 3;
}

impl fmt::Display for Prim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip_and_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for &p in Prim::ALL {
            assert!(seen.insert(p.name()), "duplicate name {}", p.name());
            assert_eq!(Prim::from_name(p.name()), Some(p));
        }
        assert_eq!(Prim::from_name("fix"), None);
    }

    #[test]
    fn nullary_prims_are_the_constructors_of_empty_values() {
        let nullary: Vec<Prim> = Prim::ALL
            .iter()
            .copied()
            .filter(|p| p.arity() == 0)
            .collect();
        assert_eq!(nullary, vec![Prim::None, Prim::Nil, Prim::Leaf]);
    }
}
