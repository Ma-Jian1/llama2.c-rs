use std::io::Write;
use std::{fs::File, time::Instant};

#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

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
            seq_len,
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
        s.x.as_mut_slice().copy_from_slice(&token_embedding[token]);

        for layer in 0..n_layers {
            rms_att.rmsnorm(&mut s.xb, &s.x, layer);

            wq.matmul_3d(&mut s.q, &s.xb, dim, dim, layer);
            wk.matmul_3d(&mut s.k, &s.xb, dim, kv_dim, layer);
            wv.matmul_3d(&mut s.v, &s.xb, dim, kv_dim, layer);

            self.rope(&mut s.q, &mut s.k, head_size, pos);

            // cache keys and values
            let idx = layer * seq_len + pos;
            Self::unchecked_mut_slice(&mut s.key_cache, kv_dim, idx).copy_from_slice(&s.k);
            Self::unchecked_mut_slice(&mut s.value_cache, kv_dim, idx).copy_from_slice(&s.v);

            self.attention(s, pos, layer);

            wo.matmul_3d(&mut s.xb2, &s.xb, dim, dim, layer);

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
        kernel::rmsnorm(&mut s.x, rms_final);

        if let Some(wcls) = wcls {
            wcls.matmul(&mut s.logits, &s.x, dim, vocab_size);
        } else {
            token_embedding.matmul(&mut s.logits, &s.x, dim, vocab_size);
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
            seq_len,
            ..
        } = self.config;
        let head_size = dim / n_heads;
        let kv_dim = n_kv_heads * head_size;
        // kv_mul q use one k head
        let kv_mul = n_heads / n_kv_heads;

        // (seq_len, kv_dim) == (seq_len, n_kv_heads, head_size)
        let layer_cached_keys = Self::unchecked_slice(&s.key_cache, seq_len * kv_dim, layer);
        let layer_cached_vals = Self::unchecked_slice(&s.value_cache, seq_len * kv_dim, layer);

        let att_lambda = |h| {
            // get the query vector for this head
            let q = Self::unchecked_slice(&s.q, head_size, h);
            // attention scores for this head
            // let att = Self::unchecked_mut_slice(&mut s.att, seq_len, h);
            let att = Self::unchecked_mut_slice(&s.att, seq_len, h);
            let mut head_k_all_pos = layer_cached_keys
                .chunks_exact(head_size)
                .skip(h / kv_mul)
                .step_by(n_kv_heads);

            // iterate over all timesteps, including the current one
            for t in 0..=pos {
                let k = head_k_all_pos.next().expect("head_k_all_pos");

                // for head_size
                let score = q.iter().zip(k.iter()).fold(0f32, |acc, (q, k)| acc + q * k);
                let score = score / (head_size as f32).sqrt();

                unsafe {
                    *att.get_unchecked_mut(t) = score;
                }
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            kernel::softmax(&mut att[..=pos]);

            let seq_cached_vals = layer_cached_vals
                .chunks_exact(head_size)
                .skip(h / kv_mul)
                .step_by(n_kv_heads);

            // let xb = Self::unchecked_mut_slice(&mut s.xb, head_size, h);
            let xb = Self::unchecked_mut_slice(&s.xb, head_size, h);
            xb.iter_mut().for_each(|v| *v = 0f32);
            for (vals, att_w) in seq_cached_vals.zip(att.iter()).take(pos + 1) {
                // aggretate timestamp to xb
                for (val, dst) in vals.iter().zip(xb.iter_mut()) {
                    *dst += att_w * val;
                }
            }
        };

        #[cfg(not(feature = "parallel"))]
        (0..n_heads).for_each(att_lambda);
        #[cfg(feature = "parallel")]
        (0..n_heads).into_par_iter().for_each(att_lambda);
    }

    fn ffn(&self, s: &mut State, layer: usize, dim: usize, hidden_dim: usize) {
        let w = &self.weights;
        w.rms_ffn.rmsnorm(&mut s.xb, &s.x, layer);

        // up scale
        w.w1.matmul_3d(&mut s.hb, &s.xb, dim, hidden_dim, layer);
        w.w3.matmul_3d(&mut s.hb2, &s.xb, dim, hidden_dim, layer);
        // silu
        kernel::silu(&mut s.hb);

        // down scale
        s.hb.iter_mut()
            .zip(s.hb2.iter())
            .for_each(|(h1, &h2)| *h1 *= h2);
        w.w2.matmul_3d(&mut s.xb, &s.hb, hidden_dim, dim, layer);
    }

    /// Treat s as 2-dimension array: [[Q; size]; x] and return &s[idx][..]
    fn unchecked_slice<Q>(s: &[Q], size: usize, idx: usize) -> &[Q] {
        let ptr = s.as_ptr();
        unsafe {
            let offset = ptr.add(idx * size);
            std::slice::from_raw_parts(offset, size)
        }
    }

    /// Treat s as 2-dimension array: [[Q; size]; x] and return &s[idx][..]
    // fn unchecked_mut_slice<Q>(s: &mut [Q], size: usize, idx: usize) -> &mut [Q] {
    //     let ptr = s.as_mut_ptr();
    //     unsafe {
    //         let offset = ptr.add(idx * size);
    //         std::slice::from_raw_parts_mut(offset, size)
    //     }
    // }
    fn unchecked_mut_slice<Q>(s: &[Q], size: usize, idx: usize) -> &mut [Q] {
        // rayon parallel need lambda implement Fn not FnMut.
        let ptr = s.as_ptr() as *mut Q;
        unsafe {
            let offset = ptr.add(idx * size);
            std::slice::from_raw_parts_mut(offset, size)
        }
    }
}
