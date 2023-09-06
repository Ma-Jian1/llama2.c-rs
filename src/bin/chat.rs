use std::io::Write;

use anyhow::Result;
use clap::Parser;

use llama2::state::State;
use llama2::Llama2Model;
use llama2::Sampler;
use llama2::{SpecialToken, Tokenizer};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "stories15M.bin")]
    config_path: String,

    #[arg(short, long, default_value = "llama2.c/tokenizer.bin")]
    tokenizer_path: String,

    #[arg(short, long, default_value = "Hi there!")]
    prompt: String,

    #[arg(
        short = 'P',
        long,
        default_value = r#"You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."#
    )]
    system_prompt: String,

    #[arg(short, long, default_value_t = 256)]
    steps: usize,

    /// [0, inf)
    #[arg(short = 'T', long, default_value_t = 0.0)]
    temperature: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut model = Llama2Model::new(&args.config_path)?;

    let tokenizer = Tokenizer::new(&args.tokenizer_path, model.config.vocab_size)?;

    let sampler = Sampler::new(args.temperature);

    let steps = args.steps;
    let steps = if steps == 0 || steps > model.config.seq_len {
        model.config.seq_len
    } else {
        steps
    };

    let mut system_prompt = args.system_prompt;
    let mut user_prompt = args.prompt;

    let (mut user_turn, mut prompt_token_idx) = (true, 0);
    let (mut token, mut next_token) = (0, 0);
    let mut prompt_tokens = Vec::<usize>::new();

    let mut state = State::new(&model.config);
    for pos in 0..steps {
        // when it is the user's turn to contribute tokens to the dialog...
        if user_turn {
            // get the (optional) system prompt at position 0
            if pos == 0 {
                if system_prompt.is_empty() {
                    read_stdin("Enter system prompt (optional): ", &mut system_prompt)?;
                } else {
                    println!("System: {}", system_prompt);
                }
                if user_prompt.is_empty() {
                    read_stdin("User: ", &mut user_prompt)?;
                } else {
                    println!("User: {}", user_prompt);
                }
            } else {
                read_stdin("User: ", &mut user_prompt)?;
            }

            // render system and user prompts into the Llama2 chat schema
            // https://huggingface.co/blog/llama2#how-to-prompt-llama-2
            // https://old.reddit.com/r/LocalLLaMA/comments/155po2p/get_llama_2_prompt_format_right/
            // https://old.reddit.com/r/LocalLLaMA/comments/1561vn5/here_is_a_practical_multiturn_llama2chat_prompt/
            let rendered_prompt = if pos == 0 && !system_prompt.is_empty() {
                format!(
                    "[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]",
                    system_prompt, user_prompt
                )
            } else {
                format!("[INST] {} [/INST]", user_prompt)
            };

            prompt_tokens = tokenizer.encode(&rendered_prompt, true, false)?;
            user_turn = false;
            prompt_token_idx = 0;

            print!("Assistant: ");
            std::io::stdout().flush()?;
        }

        // determine the token to pass into the transformer next
        if prompt_token_idx < prompt_tokens.len() {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[prompt_token_idx];
            prompt_token_idx += 1;
        } else {
            // otherwise use the next token sampled from previous turn
            token = next_token;
        }

        model.forward(&mut state, token, pos);

        if prompt_token_idx < prompt_tokens.len() {
            continue;
        }

        next_token = sampler.sample(&mut state.logits);

        if next_token == SpecialToken::Eos as usize {
            println!();
            // EOS (=2) token ends the Assistant turn
            user_turn = true;
        } else {
            // the Assistant is responding, so print its output
            print!(
                "{}",
                // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
                tokenizer.decode(next_token, token == SpecialToken::Bos as usize)?
            );
            std::io::stdout().flush()?;
        }
    }
    println!();

    Ok(())
}

fn read_stdin(prompt: &str, buffer: &mut String) -> Result<()> {
    print!("{}", prompt);
    std::io::stdout().flush()?;
    std::io::stdin().read_line(buffer)?;
    buffer.truncate(buffer.trim_end().len());
    Ok(())
}
