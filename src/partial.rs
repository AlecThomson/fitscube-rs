//! Output files that only appear once complete.
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::Result;

/// A file beside the output that is deleted on drop unless renamed onto it,
/// so a failed write leaves nothing behind.
pub(crate) struct PartialFile {
    pub path: PathBuf,
    done: bool,
}

impl PartialFile {
    /// Reserve an empty file beside `target`.
    pub fn create(target: &Path) -> Result<Self> {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        let name = target
            .file_name()
            .map_or_else(|| "cube".into(), |n| n.to_string_lossy());
        let path = target.with_file_name(format!(
            ".{name}.{}-{}.part",
            std::process::id(),
            COUNT.fetch_add(1, Ordering::Relaxed)
        ));
        File::create_new(&path)?;
        Ok(Self { path, done: false })
    }

    pub fn persist(mut self, target: &Path) -> Result<()> {
        std::fs::rename(&self.path, target)?;
        self.done = true;
        Ok(())
    }
}

impl Drop for PartialFile {
    fn drop(&mut self) {
        if !self.done {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_file_is_removed_unless_persisted() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("cube.fits");
        let partial = PartialFile::create(&target).unwrap();
        let path = partial.path.clone();
        assert!(path.exists());
        drop(partial);
        assert!(!path.exists());

        let partial = PartialFile::create(&target).unwrap();
        partial.persist(&target).unwrap();
        assert!(target.exists());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }
}
