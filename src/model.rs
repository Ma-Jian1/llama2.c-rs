use std::io::Write;
use std::{fs::File, time::Instant};

use crate::tokenizer::{SpecialToken, Tokenizer};
use crate::Result;
use crate::{config::Config, state::State, weights::Weights};
use crate::{kernel, Sampler};

pub struct Llama2Model {
    pub config: Config,
    weights: Weights,
}

impl Llama2Model {
    pub fn new(checkpoint_path: &str) -> Result<Self> {
        let mut file = File::open(checkpoint_path)?;

        let config = Config::from_reader(&mut file)?;

        let weights = Weights::from_reader(&mut file, &config)?;

        Ok(Self { config, weights })
    }

    pub fn generate(
        &mut self,
        tokenizer: &Tokenizer,
        prompt: &str,
        steps: usize,
        sampler: &Sampler,
    ) -> Result<()> {
        let mut state = State::new(&self.config);

        let prompt_tokens = tokenizer.encode(prompt, true, false)?;
        let num_prompt_tokens = prompt_tokens.len();

        let mut benches = vec![];
        let ts = Instant::now();

        let mut token = prompt_tokens[0];
        for pos in 0..steps {
            self.forward(&mut state, token, pos);

            let next_token = if pos < num_prompt_tokens - 1 {
                prompt_tokens[pos + 1]
            } else {
                sampler.sample(&mut state.logits)
            };

            // data-dependent terminating condition: the BOS (=1) token delimits sequences
            if next_token == SpecialToken::Bos as usize {
                break;
            }

            // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
            print!(
                "{}",
                tokenizer.decode(next_token, token == SpecialToken::Bos as usize)?
            );
            std::io::stdout().flush()?;

            token = next_token;

            let speed = pos as f32 / ts.elapsed().as_secs_f32();
            benches.push(speed);
        }

        let speed = benches.iter().sum::<f32>() / benches.len() as f32;
        println!("\n{:.3} tok/s, total tokens: {}", speed, benches.len());

        Ok(())
    }

    pub fn forward(&mut self, s: &mut State, token: usize, pos: usize) {
        let Config {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            ..
        } = self.config;
        let head_size = dim / n_heads;
        let kv_dim = n_kv_heads * head_size;

        let Weights {
            token_embedding,
            rms_att,
            wq,
            wk,
            wv,
            wo,
            rms_final,
            wcls,
            ..
        } = &self.weights;

        // copy current token embedding
        // s.x.copy_from_slice(&token_embedding[0][token * dim..(token + 1) * dim]);
        s.x.copy_from_slice(token_embedding.unchecked_slice(dim, token));

        for layer in 0..n_layers {
            rms_att.rmsnorm(&mut s.xb, &s.x, layer);

            wq.matmul(&mut s.q, &s.xb, dim, dim, layer);
            wk.matmul(&mut s.k, &s.xb, dim, kv_dim, layer);
            wv.matmul(&mut s.v, &s.xb, dim, kv_dim, layer);

            self.rope(&mut s.q, &mut s.k, head_size, pos);

            // cache keys and values
            s.key_cache[(layer, pos)].copy_from_slice(&s.k);
            s.value_cache[(layer, pos)].copy_from_slice(&s.v);

            self.attention(s, pos, layer);

            wo.matmul(&mut s.xb2, &s.xb, dim, dim, layer);

            // post attention residual
            s.x.iter_mut()
                .zip(s.xb2.iter())
                .for_each(|(dst, src)| *dst += *src);

            self.ffn(s, layer, dim, hidden_dim);

            // post ffn residual
            s.x.iter_mut()
                .zip(s.xb.iter())
                .for_each(|(dst, src)| *dst += *src);
        }

        // last rmsnorm
        rms_final.rmsnorm(&mut s.xb, &s.x, 0);

        if let Some(wcls) = wcls {
            wcls.matmul(&mut s.logits, &s.xb, dim, vocab_size, 0);
        } else {
            token_embedding.matmul(&mut s.logits, &s.xb, dim, vocab_size, 0);
        }
    }

    fn rope(&self, q: &mut [f32], k: &mut [f32], head_size: usize, pos: usize) {
        // q may not equal to k, but they all should be divided by head_size
        kernel::rope(q, head_size, pos);
        kernel::rope(k, head_size, pos);
    }

    fn attention(&self, s: &mut State, pos: usize, layer: usize) {
        let Config {
            dim,
            n_heads,
            n_kv_heads,
            ..
        } = self.config;
        let head_size = dim / n_heads;
        // kv_mul q use one k head
        let kv_mul = n_heads / n_kv_heads;

        // (seq_len, kv_dim) == (seq_len, n_kv_heads, head_size)
        let layer_cached_keys = &s.key_cache[layer];
        let layer_cached_vals = &s.value_cache[layer];

        (0..n_heads).for_each(|h| {
            // get the query vector for this head
            let q = s.q.unchecked_slice(head_size, h);
            // attention scores for this head
            let att = &mut s.att[h];
            att.iter_mut()
                .zip(
                    layer_cached_keys
                        .chunks_exact(head_size)
                        .skip(h / kv_mul)
                        .step_by(n_kv_heads),
                )
                // iterate over all timesteps, including the current one
                .take(pos + 1)
                .for_each(|(att, k)| {
                    // for head_size
                    let score = q.iter().zip(k.iter()).fold(0f32, |acc, (q, k)| acc + q * k);
                    let score = score / (head_size as f32).sqrt();

                    *att = score;
                });

            // softmax the scores to get attention weights, from 0..pos inclusively
            kernel::softmax(&mut att[..=pos]);

            let xb = s.xb.unchecked_mut_slice(head_size, h);
            xb.iter_mut().for_each(|v| *v = 0f32);
            att.iter()
                .zip(
                    layer_cached_vals
                        .chunks_exact(head_size)
                        .skip(h / kv_mul)
                        .step_by(n_kv_heads),
                )
                .take(pos + 1)
                .for_each(|(att, vals)| {
                    xb.iter_mut().zip(vals.iter()).for_each(|(xb, val)| {
                        *xb += att * val;
                    })
                });
        })
    }

    fn ffn(&self, s: &mut State, layer: usize, dim: usize, hidden_dim: usize) {
        let w = &self.weights;
        w.rms_ffn.rmsnorm(&mut s.xb, &s.x, layer);

        // up scale
        w.w1.matmul(&mut s.hb, &s.xb, dim, hidden_dim, layer);
        w.w3.matmul(&mut s.hb2, &s.xb, dim, hidden_dim, layer);
        // silu
        kernel::silu(&mut s.hb);

        // down scale
        s.hb.iter_mut()
            .zip(s.hb2.iter())
            .for_each(|(h1, &h2)| *h1 *= h2);
        w.w2.matmul(&mut s.xb, &s.hb, hidden_dim, dim, layer);
    }
}
