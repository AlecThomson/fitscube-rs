//! Gzip a finished cube, as fitscube's `--compress` does.
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use flate2::Compression;
use flate2::write::GzEncoder;

use crate::error::Result;
use crate::partial::PartialFile;

/// Gzip `path` to `path.gz` and delete `path`. Returns the new path.
pub fn compress_cube(path: &Path) -> Result<PathBuf> {
    let mut gz_name = path.as_os_str().to_owned();
    gz_name.push(".gz");
    let gz_path = PathBuf::from(gz_name);
    tracing::info!("Compressing {} to {}", path.display(), gz_path.display());

    let partial = PartialFile::create(&gz_path)?;
    let mut src = BufReader::new(File::open(path)?);
    // Python's gzip.open defaults to level 9
    let mut gz = GzEncoder::new(
        BufWriter::new(File::create(&partial.path)?),
        Compression::best(),
    );
    std::io::copy(&mut src, &mut gz)?;
    gz.finish()?
        .into_inner()
        .map_err(|e| e.into_error())?
        .sync_all()?;
    partial.persist(&gz_path)?;
    std::fs::remove_file(path)?;
    Ok(gz_path)
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use flate2::read::GzDecoder;

    use super::*;

    #[test]
    fn round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cube.fits");
        let data: Vec<u8> = (0..10_000u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        let gz = compress_cube(&path).unwrap();
        assert_eq!(gz, dir.path().join("cube.fits.gz"));
        assert!(!path.exists());
        let mut back = Vec::new();
        GzDecoder::new(File::open(gz).unwrap())
            .read_to_end(&mut back)
            .unwrap();
        assert_eq!(back, data);
    }
}
