//! Progress bars and spinners for long-running CLI work.
//!
//! Mirrors the helpers in the sibling `convolve-rs` tool. These are opt-in: the
//! library only builds them when a caller sets `progress` on its options
//! (the CLI does; the Python bindings leave it off so importing the module
//! stays quiet). The work loops in [`crate::combine`] drive them.
use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};

/// A determinate bar for `total` items (e.g. one tick per written channel).
pub fn progress_bar(total: u64) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );
    // Steady tick animates the `{spinner}` even when `pos` is not advancing, so
    // idle-but-busy phases (e.g. the parallel readers warming up before the
    // first plane lands) still show live activity rather than a frozen bar.
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// An indeterminate spinner for blocking phases with no item count (e.g.
/// solving for a common bounding box). The caller clears it via
/// `finish_and_clear` when the work completes.
pub fn spinner(msg: impl Into<String>) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed}] {msg}")
            .unwrap(),
    );
    pb.set_message(msg.into());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}
