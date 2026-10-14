//! Progress bars for the CLI. The library draws them only when asked to
//! (`CombineOptions::progress`).
use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};

/// A bar for `total` items.
pub fn progress_bar(total: u64) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );
    // Keep the spinner moving before the first plane lands
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// A spinner for work with no item count.
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
