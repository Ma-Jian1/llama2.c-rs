use std::fs::File;

use anyhow::Result;
use clap::Parser;

use llama2::{config::Config, weights::TransformerWeights};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "stories15M.bin")]
    config_path: String,

    #[arg(short, long, default_value = "")]
    prompt: String,

    /// [0, inf)
    #[arg(short, long, default_value_t = 1.0)]
    temperature: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("{args:#?}");

    let mut file = File::open(args.config_path)?;

    let config = Config::from_reader(&mut file)?;
    println!("{config:#?}");

    let weights = TransformerWeights::from_reader(&mut file, &config)?;
    // println!("{weights:#?}");

    Ok(())
}
