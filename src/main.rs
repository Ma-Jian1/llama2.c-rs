pub mod model;
pub mod tokenizer;

use anyhow::Result;
use clap::Parser;
use llama2::tokenizer::Tokenizer;

use crate::model::Llama2Model;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "stories15M.bin")]
    config_path: String,

    #[arg(short, long, default_value = "tokenizer.bin")]
    tokenizer_path: String,

    #[arg(short, long, default_value = "")]
    prompt: String,

    /// [0, inf)
    #[arg(short = 'T', long, default_value_t = 1.0)]
    temperature: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("{args:#?}");

    let model = Llama2Model::new(&args.config_path)?;

    let tokenizer = Tokenizer::new(&args.tokenizer_path, model.config.vocab_size)?;

    Ok(())
}
