//! Records what the benchmark is built from, to report with its numbers:
//! `PANETIERE_REVISION` is the commit, with `-dirty` if tracked files have
//! uncommitted changes, or `unknown` outside a git checkout, and
//! `PANETIERE_RUSTC` is the compiler's version.

use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Tell cargo to rerun this script when `path` changes. Only for paths that
/// exist: cargo reruns the script on every build for a missing one.
fn watch(path: &str) {
    if Path::new(path).exists() {
        println!("cargo:rerun-if-changed={path}");
    }
}

fn main() {
    let revision = match git(&["rev-parse", "--short=12", "HEAD"]) {
        Some(rev) => {
            let dirty = git(&[
                "--no-optional-locks",
                "status",
                "--porcelain",
                "--untracked-files=no",
            ])
            .is_some_and(|s| !s.is_empty());
            if dirty {
                format!("{rev}-dirty")
            } else {
                rev
            }
        }
        None => "unknown".into(),
    };
    println!("cargo:rustc-env=PANETIERE_REVISION={revision}");

    // Cargo tells every build script which compiler it uses. A new compiler
    // rebuilds this script, so the version stays current.
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version = Command::new(&rustc)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=PANETIERE_RUSTC={version}");

    // Rerun when the checked-out commit or a tracked file changes: the
    // repository's HEAD, index and refs, and every top-level entry with
    // tracked files, so that `-dirty` matches `git status`. A rerun only
    // rebuilds this crate, not the library.
    watch("build.rs");
    if let Some(dir) = git(&["rev-parse", "--git-dir"]) {
        watch(&format!("{dir}/HEAD"));
        watch(&format!("{dir}/index"));
    }
    if let Some(dir) = git(&["rev-parse", "--git-common-dir"]) {
        watch(&format!("{dir}/refs"));
        watch(&format!("{dir}/packed-refs"));
    }
    if let (Some(top), Some(files)) = (
        git(&["rev-parse", "--show-toplevel"]),
        git(&["ls-files", "--full-name", ":/"]),
    ) {
        let entries: BTreeSet<&str> = files.lines().filter_map(|f| f.split('/').next()).collect();
        for entry in entries {
            watch(&format!("{top}/{entry}"));
        }
    }
}
