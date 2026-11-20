//! `dc-compare <spec.json> [--json <report.json>] [--commit <id>]`
//!
//! Analyses an arm comparison of decision D33 and prints its report. With
//! `--json` it also writes the report as JSON. `--commit` names the commit
//! of dc-compare in the report, for the results log.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use dc_compare::compare::{analyse, Spec};
use dc_compare::render;

fn usage() -> ExitCode {
    eprintln!("usage: dc-compare <spec.json> [--json <report.json>] [--commit <id>]");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut spec_path: Option<PathBuf> = None;
    let mut json: Option<PathBuf> = None;
    let mut commit: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--json" => match args.next() {
                Some(p) => json = Some(PathBuf::from(p)),
                None => return usage(),
            },
            "--commit" => match args.next() {
                Some(c) => commit = Some(c),
                None => return usage(),
            },
            "-h" | "--help" => return usage(),
            _ if spec_path.is_none() && !a.starts_with('-') => spec_path = Some(PathBuf::from(a)),
            _ => return usage(),
        }
    }
    let Some(spec_path) = spec_path else {
        return usage();
    };
    let text = match std::fs::read_to_string(&spec_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("dc-compare: cannot read {}: {e}", spec_path.display());
            return ExitCode::FAILURE;
        }
    };
    let spec: Spec = match serde_json::from_str(&text) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("dc-compare: cannot parse {}: {e}", spec_path.display());
            return ExitCode::FAILURE;
        }
    };
    let base = spec_path.parent().unwrap_or(Path::new("."));
    let report = match analyse(&spec, base) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("dc-compare: {e}");
            return ExitCode::FAILURE;
        }
    };
    print!("{}", render::text(&report, commit.as_deref()));
    if let Some(path) = json {
        let mut value = match serde_json::to_value(&report) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("dc-compare: cannot write the report as JSON: {e}");
                return ExitCode::FAILURE;
            }
        };
        if let (Some(obj), Some(c)) = (value.as_object_mut(), &commit) {
            obj.insert("commit".to_owned(), serde_json::Value::String(c.clone()));
        }
        let written = serde_json::to_string_pretty(&value)
            .map_err(|e| e.to_string())
            .and_then(|t| std::fs::write(&path, t + "\n").map_err(|e| e.to_string()));
        if let Err(e) = written {
            eprintln!("dc-compare: cannot write {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}
