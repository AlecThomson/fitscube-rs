//! Header keyword reads that tell an absent key apart from a read failure.
use fitsio::FitsFile;
use fitsio::errors::Error as FitsioError;
use fitsio::hdu::FitsHdu;
use fitsio::headers::ReadsKey;

use crate::error::Result;

/// cfitsio `KEY_NO_EXIST`.
const KEY_NO_EXIST: i32 = 202;

/// Read `key`, or `None` if the header does not have it. Any other failure,
/// such as a value of the wrong type, is an error.
pub(crate) fn read_opt_key<T: ReadsKey>(
    hdu: &FitsHdu,
    fptr: &mut FitsFile,
    key: &str,
) -> Result<Option<T>> {
    match hdu.read_key::<T>(fptr, key) {
        Ok(v) => Ok(Some(v)),
        Err(FitsioError::Fits(e)) if e.status == KEY_NO_EXIST => Ok(None),
        Err(e) => Err(e.into()),
    }
}
