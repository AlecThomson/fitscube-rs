//! Image headers built from the cards of any image HDU, plain or tile-compressed.
use std::ffi::{CStr, c_char, c_int};

use atfits_rs::{is_scaling_keyword, is_structural_keyword, key_i64, update_key_i64};
use fitsio::FitsFile;
use fitsio::errors::check_status;
use fitsio::sys::{self, LONGLONG};

use crate::error::{FitsCubeError, Result};

/// The cards of the current header that describe the image rather than its
/// storage, plain or compressed. BSCALE, BZERO and BLANK are kept if `scaling`.
pub(crate) fn user_cards(fptr: &mut FitsFile, scaling: bool) -> Result<Vec<u8>> {
    let mut status = 0;
    let mut nkeys: c_int = 0;
    unsafe { sys::ffghsp(fptr.as_raw(), &mut nkeys, std::ptr::null_mut(), &mut status) };
    check_status(status)?;
    let mut cards = Vec::new();
    let mut card = [0 as c_char; 81];
    for i in 1..=nkeys {
        unsafe { sys::ffgrec(fptr.as_raw(), i, card.as_mut_ptr(), &mut status) };
        check_status(status)?;
        let text = unsafe { CStr::from_ptr(card.as_ptr()) }.to_bytes();
        let name = card_name(text);
        if keep_card(name) || (scaling && is_scaling_keyword(name)) {
            cards.extend_from_slice(&pad_card(text));
        }
    }
    Ok(cards)
}

/// A new in-memory primary image of `bitpix` and `dims` with the cards of the
/// current image HDU of `fptr`, which may be tile-compressed.
pub(crate) fn plain_header(fptr: &mut FitsFile, bitpix: i64, dims: &[usize]) -> Result<FitsFile> {
    // Not CFITSIO's ffcnvthdr2str, which writes the whole image into memory
    let cards = user_cards(fptr, bitpix > 0)?;
    let mut status = 0;
    let mut raw: *mut sys::fitsfile = std::ptr::null_mut();
    unsafe { sys::ffinit(&mut raw, c"mem://".as_ptr(), &mut status) };
    check_status(status)?;
    let mut header = unsafe { FitsFile::from_raw(raw, fitsio::FileOpenMode::READWRITE)? };
    let mut naxes: Vec<LONGLONG> = vec![1; dims.len()];
    unsafe {
        sys::ffcrimll(
            header.as_raw(),
            bitpix as c_int,
            naxes.len() as c_int,
            naxes.as_mut_ptr(),
            &mut status,
        )
    };
    check_status(status)?;
    for card in cards.chunks(80) {
        let card =
            std::ffi::CString::new(card).map_err(|e| FitsCubeError::Atfits(e.to_string()))?;
        unsafe { sys::ffprec(header.as_raw(), card.as_ptr(), &mut status) };
        check_status(status)?;
    }
    if bitpix > 0
        && let Some(blank) = key_i64(fptr, "ZBLANK")?
    {
        update_key_i64(&mut header, "BLANK", blank)?;
    }
    // Re-read the header so CFITSIO applies any copied BSCALE/BZERO/BLANK
    unsafe { sys::ffrdef(header.as_raw(), &mut status) };
    check_status(status)?;
    Ok(header)
}

/// Whether a card describes the image rather than how it is stored.
pub(crate) fn keep_card(name: &str) -> bool {
    let reserved = [
        "ZIMAGE", "ZSIMPLE", "ZEXTEND", "ZBITPIX", "ZCMPTYPE", "ZQUANTIZ", "ZDITHER0", "ZBLANK",
        "ZSCALE", "ZZERO", "ZTENSION", "ZPCOUNT", "ZGCOUNT", "ZHECKSUM", "ZDATASUM", "XTENSION",
        "EXTNAME", "TFIELDS", "THEAP", "CHECKSUM", "DATASUM",
    ];
    let numbered = [
        "ZNAXIS", "ZTILE", "ZNAME", "ZVAL", "TTYPE", "TFORM", "TUNIT", "TDIM", "TNULL", "TSCAL",
        "TZERO", "TDISP",
    ];
    !(is_structural_keyword(name)
        || is_scaling_keyword(name)
        || reserved.contains(&name)
        || numbered.iter().any(|p| {
            name.strip_prefix(p)
                .is_some_and(|n| n.bytes().all(|c| c.is_ascii_digit()))
        }))
}

/// Keyword name of a card: the text before the first space or `=`.
fn card_name(card: &[u8]) -> &str {
    let end = card
        .iter()
        .position(|&b| b == b' ' || b == b'=')
        .unwrap_or(card.len());
    std::str::from_utf8(&card[..end]).unwrap_or("")
}

pub(crate) fn pad_card(text: &[u8]) -> [u8; 80] {
    let mut card = [b' '; 80];
    let n = text.len().min(80);
    card[..n].copy_from_slice(&text[..n]);
    card
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_user_cards_only() {
        assert!(keep_card("BUNIT"));
        assert!(keep_card("HISTORY"));
        assert!(keep_card("ZOOM"));
        assert!(!keep_card("NAXIS3"));
        assert!(!keep_card("ZNAXIS2"));
        assert!(!keep_card("BSCALE"));
        assert!(!keep_card("CHECKSUM"));
    }
}
