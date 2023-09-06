use anyhow::Result;
use clap::Parser;

use llama2::Llama2Model;
use llama2::Sampler;
use llama2::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "stories15M.bin")]
    config_path: String,

    #[arg(short, long, default_value = "llama2.c/tokenizer.bin")]
    tokenizer_path: String,

    #[arg(short, long, default_value = "")]
    prompt: String,

    #[arg(short, long, default_value_t = 256)]
    steps: usize,

    /// [0, inf)
    #[arg(short = 'T', long, default_value_t = 0.0)]
    temperature: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut model = Llama2Model::new(&args.config_path)?;

    #[cfg(feature = "parallel")]
    {
        let cpus = num_cpus::get();
        let active_cpus = cpus.max(1).min(model.config.n_heads); // use 75% of available cores
        println!("\n--> [Running Inference on {} CPUs]\n", active_cpus);

        rayon::ThreadPoolBuilder::new()
            .num_threads(active_cpus)
            .build_global()
            .unwrap();
    }

    let tokenizer = Tokenizer::new(&args.tokenizer_path, model.config.vocab_size)?;

    let sampler = Sampler::new(args.temperature);

    let steps = args.steps;
    let steps = if steps == 0 || steps > model.config.seq_len {
        model.config.seq_len
    } else {
        steps
    };

    model.generate(&tokenizer, &args.prompt, steps, &sampler)?;

    Ok(())
}
