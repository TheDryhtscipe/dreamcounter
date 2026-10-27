//! The table between library abstractions ([`LibId`]) and their Stitch names.

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use dc_core::{LibId, Prim};

use crate::codec::{AND, IF, INT_PREFIX, LET, OR};

/// A one-to-one table between library ids and Stitch token names.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LibNames {
    by_id: BTreeMap<LibId, String>,
    by_name: HashMap<String, LibId>,
}

/// Why a name could not be added to a [`LibNames`] table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NameError {
    /// The name cannot be a Stitch token, or would be read as something else.
    Illegal {
        /// The name offered.
        name: String,
        /// Why it was refused.
        reason: &'static str,
    },
    /// The name is already given to another id.
    NameTaken {
        /// The name offered.
        name: String,
        /// The id that has it.
        holder: LibId,
    },
    /// The id already has a name.
    IdNamed {
        /// The id offered.
        id: LibId,
        /// Its existing name.
        name: String,
    },
}

impl fmt::Display for NameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NameError::Illegal { name, reason } => {
                write!(f, "'{name}' cannot name an abstraction: {reason}")
            }
            NameError::NameTaken { name, holder } => {
                write!(f, "'{name}' already names library {}", holder.0)
            }
            NameError::IdNamed { id, name } => {
                write!(f, "library {} is already named '{name}'", id.0)
            }
        }
    }
}

/// Why `name` cannot be an abstraction name, or `None` if it can.
fn illegal_reason(name: &str) -> Option<&'static str> {
    if name.is_empty() {
        return Some("it is empty");
    }
    if name
        .chars()
        .any(|c| c.is_whitespace() || c == '(' || c == ')')
    {
        return Some("Stitch splits tokens on whitespace and parentheses");
    }
    if name.starts_with(['$', '#', '&']) {
        return Some("Stitch reads '$', '#' and '&' prefixes as variables");
    }
    if name == "??" {
        return Some("Stitch prints holes as '??'");
    }
    if name == "lam" || name == "lambda" || name.starts_with("lam_") || name.starts_with("lambda_")
    {
        return Some("Stitch reads it as a lambda");
    }
    if [LET, IF, AND, OR].contains(&name) {
        return Some("it is a reserved token for a special form");
    }
    if name.starts_with(INT_PREFIX) {
        return Some("the 'int_' prefix is reserved for integer literals");
    }
    if matches!(name, "true" | "false" | "unit") || Prim::from_name(name).is_some() {
        return Some("it is a literal or primitive name");
    }
    None
}

impl std::error::Error for NameError {}

impl LibNames {
    /// An empty table.
    pub fn new() -> LibNames {
        LibNames::default()
    }

    /// Names `id` as `name`.
    ///
    /// The name must be a single Stitch token that Stitch and the decoder would not
    /// read as anything else: not a variable, parameter, lambda, hole, literal,
    /// primitive or reserved token, and not starting with `&` (Stitch's symbol
    /// variable prefix). Each id has at most one name and each name one id.
    pub fn insert(&mut self, id: LibId, name: impl Into<String>) -> Result<(), NameError> {
        let name = name.into();
        if let Some(reason) = illegal_reason(&name) {
            return Err(NameError::Illegal { name, reason });
        }
        if let Some(&holder) = self.by_name.get(&name) {
            return Err(NameError::NameTaken { name, holder });
        }
        if let Some(existing) = self.by_id.get(&id) {
            return Err(NameError::IdNamed {
                id,
                name: existing.clone(),
            });
        }
        self.by_name.insert(name.clone(), id);
        self.by_id.insert(id, name);
        Ok(())
    }

    /// The table naming `LibId(i)` as `{prefix}{i}` for `i` in `0..count`, which
    /// matches Stitch's own naming (`fn_0`, `fn_1`, ...) when `prefix` is
    /// Stitch's `--abstraction-prefix`.
    pub fn sequential(prefix: &str, count: u32) -> Result<LibNames, NameError> {
        let mut t = LibNames::new();
        for i in 0..count {
            t.insert(LibId(i), format!("{prefix}{i}"))?;
        }
        Ok(t)
    }

    /// The name of `id`, if it has one.
    pub fn name(&self, id: LibId) -> Option<&str> {
        self.by_id.get(&id).map(String::as_str)
    }

    /// The id named `name`, if any.
    pub fn id(&self, name: &str) -> Option<LibId> {
        self.by_name.get(name).copied()
    }

    /// The number of named abstractions.
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_map_both_ways() {
        let mut t = LibNames::new();
        t.insert(LibId(3), "fn_3").unwrap();
        t.insert(LibId(0), "my_helper").unwrap();
        assert_eq!(t.name(LibId(3)), Some("fn_3"));
        assert_eq!(t.id("fn_3"), Some(LibId(3)));
        assert_eq!(t.id("my_helper"), Some(LibId(0)));
        assert_eq!(t.name(LibId(1)), None);
        assert_eq!(t.id("fn_0"), None);
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn sequential_matches_stitch_naming() {
        let t = LibNames::sequential("fn_", 3).unwrap();
        assert_eq!(t.len(), 3);
        for i in 0..3 {
            assert_eq!(t.name(LibId(i)), Some(format!("fn_{i}").as_str()));
        }
    }

    #[test]
    fn duplicates_are_refused() {
        let mut t = LibNames::new();
        t.insert(LibId(0), "fn_0").unwrap();
        assert_eq!(
            t.insert(LibId(1), "fn_0"),
            Err(NameError::NameTaken {
                name: "fn_0".into(),
                holder: LibId(0)
            })
        );
        assert_eq!(
            t.insert(LibId(0), "fn_9"),
            Err(NameError::IdNamed {
                id: LibId(0),
                name: "fn_0".into()
            })
        );
        // A refused insert leaves the table unchanged.
        assert_eq!(t.len(), 1);
        assert_eq!(t.id("fn_9"), None);
    }

    #[test]
    fn names_that_would_read_as_something_else_are_refused() {
        for name in [
            "", "a b", "a(b", "a)b", "$0", "#0", "&x", "??", "lam", "lambda", "lam_1", "lambda_2",
            "let_", "if_", "and_", "or_", "int_3", "int_x", "add", "foldl", "true", "false",
            "unit",
        ] {
            let mut t = LibNames::new();
            match t.insert(LibId(0), name) {
                Err(NameError::Illegal { name: n, .. }) => assert_eq!(n, name),
                other => panic!("{name:?} was not refused: {other:?}"),
            }
            assert!(t.is_empty());
        }
        assert!(matches!(
            LibNames::sequential("int_", 1),
            Err(NameError::Illegal { .. })
        ));
    }
}
