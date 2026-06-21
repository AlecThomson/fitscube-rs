//! `fitscubers` command-line interface.
//!
//! Two subcommands mirror the original package:
//! * `fitscubers combine` — combine single-plane images into a cube.
//! * `fitscubers extract` — extract one plane from a cube.
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use tracing::info;
use tracing_subscriber::EnvFilter;

use fitscube_rs::combine::{CombineOptions, combine_fits};
use fitscube_rs::extract::{ExtractOptions, extract_plane_from_cube};

#[derive(Parser)]
#[command(
    name = "fitscubers",
    about = "Combine single-frequency/single-time FITS images into a cube, or extract a plane",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Combine FITS files into a cube (in frequency or time order).
    Combine(CombineArgs),
    /// Extract a plane (channel or timestep) from a FITS cube.
    Extract(ExtractArgs),
}

#[derive(Args)]
struct CombineArgs {
    /// FITS files to combine (in frequency or time order).
    #[arg(required = true, num_args = 1..)]
    file_list: Vec<PathBuf>,
    /// Output FITS cube path.
    out_cube: PathBuf,
    /// Overwrite the output file if it exists.
    #[arg(short, long)]
    overwrite: bool,
    /// Try to create a cube with evenly spaced frequencies/times (blank gaps).
    #[arg(long)]
    create_blanks: bool,
    /// Build a time-domain cube (DATE-OBS) instead of frequency (FREQ/REFFREQ).
    #[arg(long)]
    time_domain: bool,
    /// File of frequencies (Hz) or times (MJD s), one per line.
    #[arg(long, group = "spec")]
    spec_file: Option<PathBuf>,
    /// Frequencies (Hz) or times (MJD s) given on the command line.
    #[arg(long, num_args = 1.., group = "spec")]
    specs: Option<Vec<f64>>,
    /// Ignore frequency/time information and just stack.
    #[arg(long, group = "spec")]
    ignore_spec: bool,
    /// Increase output verbosity (repeatable).
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbosity: u8,
    /// Maximum number of in-flight planes (concurrency bound).
    #[arg(long)]
    max_workers: Option<usize>,
    /// Trim blank padding around the data using a common bounding box.
    #[arg(long)]
    bounding_box: bool,
    /// Set pixels whose values are exactly zero to NaN.
    #[arg(long)]
    invalidate_zeros: bool,
    /// Output float precision in bits (32 or 64). Defaults to the input.
    #[arg(long, value_parser = ["32", "64"])]
    floating: Option<String>,
}

#[derive(Args)]
struct ExtractArgs {
    /// The cube to extract a plane from.
    fits_cube: PathBuf,
    /// Frequency channel index to extract.
    #[arg(long, group = "idx")]
    channel_index: Option<usize>,
    /// Timestep index to extract.
    #[arg(long, group = "idx")]
    time_index: Option<usize>,
    /// HDU index of the cube data.
    #[arg(long, default_value_t = 0)]
    hdu_index: usize,
    /// Increase output verbosity (repeatable).
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbosity: u8,
    /// Overwrite the output file if it exists.
    #[arg(long)]
    overwrite: bool,
    /// Output path. Generated from the cube name if omitted.
    #[arg(long)]
    output_path: Option<PathBuf>,
}

fn init_logging(verbosity: u8) {
    let level = match verbosity {
        0 => "warn",
        1 => "info",
        _ => "debug",
    };
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("fitscube={level},fitscube_rs={level}")));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}

fn run_combine(args: CombineArgs) -> Result<()> {
    init_logging(args.verbosity);

    if !args.overwrite && args.out_cube.exists() {
        anyhow::bail!(
            "Output file {} already exists. Use --overwrite to overwrite.",
            args.out_cube.display()
        );
    }

    let options = CombineOptions {
        spec_file: args.spec_file,
        spec_list: args.specs,
        ignore_spec: args.ignore_spec,
        create_blanks: args.create_blanks,
        overwrite: args.overwrite,
        max_workers: args.max_workers,
        time_domain_mode: args.time_domain,
        bounding_box: args.bounding_box,
        invalidate_zeros: args.invalidate_zeros,
        float_length: args.floating.as_deref().map(|s| s.parse().unwrap()),
        // CLI: show progress bars on stderr.
        progress: true,
    };

    let specs = combine_fits(&args.file_list, &args.out_cube, &options)
        .context("combining FITS files into a cube")?;

    info!("Written cube to {}", args.out_cube.display());

    // Write the axis values alongside the cube, as the original CLI does.
    let unit = if args.time_domain { "s" } else { "Hz" };
    let specs_file = args.out_cube.with_extension(format!("specs_{unit}.txt"));
    let body: String = specs.iter().map(|v| format!("{v}\n")).collect::<String>();
    std::fs::write(&specs_file, body)
        .with_context(|| format!("writing spec values to {}", specs_file.display()))?;
    info!("Written axis values to {}", specs_file.display());
    Ok(())
}

fn run_extract(args: ExtractArgs) -> Result<()> {
    init_logging(args.verbosity);

    let options = ExtractOptions {
        hdu_index: args.hdu_index,
        channel_index: args.channel_index,
        time_index: args.time_index,
        overwrite: args.overwrite,
        output_path: args.output_path,
    };
    let spin = fitscube_rs::progress::spinner("extracting plane from cube");
    let out =
        extract_plane_from_cube(&args.fits_cube, &options).context("extracting plane from cube")?;
    spin.finish_and_clear();
    info!("Written plane to {}", out.display());
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Combine(args) => run_combine(args),
        Command::Extract(args) => run_extract(args),
    }
}
