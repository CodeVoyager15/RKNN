use std::{
    collections::HashSet,
    env,
    path::{Path, PathBuf},
};

const LIBRARY_NAMES: &[&str] = &["librknnrt.so", "librknnmrt.so"];

/// Find librknnrt.so or librknnmrt.so in usual locations
///
/// searches in default directories, LD_LIBRARY_PATH, and $HOME/.local/lib if available
///
pub fn find_rknn_library() -> impl Iterator<Item = PathBuf> {
    let mut seen = HashSet::new();

    let default_dirs = vec![
        "/usr/lib",
        "/usr/local/lib",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/local/lib/aarch64-linux-gnu",
        "/lib",
        "/lib/aarch64-linux-gnu",
        "/opt/lib",
    ];

    let mut search_dirs: Vec<PathBuf> = default_dirs.into_iter().map(PathBuf::from).collect();

    if let Some(ld_path) = env::var_os("LD_LIBRARY_PATH") {
        search_dirs.extend(env::split_paths(&ld_path));
    }

    if let Some(home) = env::var_os("HOME") {
        search_dirs.push(Path::new(&home).join(".local/lib"));
    }

    search_dirs
        .into_iter()
        .flat_map(|dir| LIBRARY_NAMES.iter().map(move |lib| dir.join(lib)))
        .filter(move |path| {
            // Deduplicate
            if seen.contains(path) {
                false
            } else {
                seen.insert(path.clone())
            }
        })
        .filter(|path| path.exists())
}
