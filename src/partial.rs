//! Output files that only appear once complete.
use std::path::Path;

use tempfile::TempPath;

use crate::error::Result;

/// An empty, uniquely named file beside `target`. It is deleted on drop unless
/// persisted onto `target`, so a failed write leaves nothing behind.
pub(crate) fn partial_file(target: &Path) -> Result<TempPath> {
    let dir = target
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = target
        .file_name()
        .map_or_else(|| "cube".into(), |n| n.to_string_lossy());
    let prefix = format!(".{name}.");
    let mut builder = tempfile::Builder::new();
    builder.prefix(&prefix).suffix(".part");
    // tempfile defaults to 0600; the output should get the usual umask mode.
    #[cfg(unix)]
    builder.permissions(std::os::unix::fs::PermissionsExt::from_mode(0o666));
    let file = builder.tempfile_in(dir)?;
    Ok(file.into_temp_path())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_file_is_removed_unless_persisted() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("cube.fits");
        let partial = partial_file(&target).unwrap();
        let path = partial.to_path_buf();
        assert!(path.exists());
        drop(partial);
        assert!(!path.exists());

        partial_file(&target).unwrap().persist(&target).unwrap();
        assert!(target.exists());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn partial_file_has_default_permissions() {
        let dir = tempfile::tempdir().unwrap();
        let plain = dir.path().join("plain");
        std::fs::File::create(&plain).unwrap();
        let partial = partial_file(&dir.path().join("cube.fits")).unwrap();
        let mode = |p: &Path| p.metadata().unwrap().permissions();
        assert_eq!(mode(&partial), mode(&plain));
    }
}
