use std::io::Write;
use std::{fs::File, time::Instant};

use crate::operator;
use crate::tokenizer::{SpecialToken, Tokenizer};
use crate::Result;
use crate::{config::Config, state::State, weights::Weights};

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
        temperature: f32,
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
                self.sample(&mut state.logits, temperature)
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

    pub fn forward(&mut self, state: &mut State, token: usize, pos: usize) {
        let State {
            x,
            xb,
            xb2,
            hb,
            hb2,
            q,
            k,
            v,
            att,
            logits,
            key_cache,
            value_cache,
        } = state;

        let Config {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len,
            shared_weights: _,
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
        x.as_mut_slice()
            .copy_from_slice(Self::unchecked_slice(token_embedding, dim, token));

        for layer in 0..n_layers {
            let rms_att_w = Self::unchecked_slice(rms_att, dim, layer);
            operator::rmsnorm(xb, rms_att_w, x);

            let wq = Self::unchecked_slice(wq, dim * dim, layer);
            let wk = Self::unchecked_slice(wk, dim * kv_dim, layer);
            let wv = Self::unchecked_slice(wv, dim * kv_dim, layer);
            operator::matmul(q, wq, xb, dim, dim);
            operator::matmul(k, wk, xb, dim, kv_dim);
            operator::matmul(v, wv, xb, dim, kv_dim);

            self.rope(q, k, head_size, pos);

            // cache keys and values
            let idx = layer * seq_len + pos;
            Self::unchecked_mut_slice(key_cache, kv_dim, idx).copy_from_slice(k);
            Self::unchecked_mut_slice(value_cache, kv_dim, idx).copy_from_slice(v);

            self.attention(
                key_cache,
                value_cache,
                q,
                xb,
                att,
                pos,
                layer,
                kv_dim,
                n_layers,
                n_heads,
                head_size,
                seq_len,
            );

            let wo = Self::unchecked_slice(wo, dim * dim, layer);
            operator::matmul(xb2, wo, xb, dim, dim);

            // post attention residual
            x.iter_mut()
                .zip(xb2.iter())
                .for_each(|(dst, src)| *dst += *src);

            self.ffn(x, xb, hb, hb2, layer, dim, hidden_dim);

            // post ffn residual
            x.iter_mut()
                .zip(xb.iter())
                .for_each(|(dst, src)| *dst += *src);
        }

        // last rmsnorm
        operator::rmsnorm_inplace(x, rms_final);

        if let Some(wcls) = wcls {
            operator::matmul(logits, wcls, x, dim, vocab_size);
        } else {
            operator::matmul(logits, token_embedding, x, dim, vocab_size);
        }
    }

    pub fn sample(&mut self, logits: &mut [f32], temperature: f32) -> usize {
        if temperature == 0.0 {
            operator::argmax(logits)
        } else {
            logits.iter_mut().for_each(|logit| *logit /= temperature);
            operator::softmax(logits);
            operator::sample(logits)
        }
    }

    fn rope(&self, q: &mut [f32], k: &mut [f32], head_size: usize, pos: usize) {
        // q may not equal to k, but they all should be divided by head_size
        operator::rope(q, head_size, pos);
        operator::rope(k, head_size, pos);
    }

    fn attention(
        &self,
        key_cache: &[f32],
        value_cache: &[f32],
        q: &[f32],
        xb: &mut [f32],
        att: &mut [f32],
        pos: usize,
        layer: usize,
        kv_dim: usize,
        n_layers: usize,
        n_heads: usize,
        head_size: usize,
        seq_len: usize,
    ) {
        // (seq_len, kv_dim)
        let layer_cached_keys = Self::unchecked_slice(key_cache, seq_len * kv_dim, layer);

        (0..n_heads).for_each(|h| {
            let q = Self::unchecked_slice(q, head_size, h);
            let xb = Self::unchecked_mut_slice(xb, head_size, h);
            let att = Self::unchecked_mut_slice(att, seq_len, h);
            let layer_cached_vals =
                Self::unchecked_slice(value_cache, n_layers * seq_len * kv_dim, 0);

            let mut head_k_all_pos = layer_cached_keys
                .chunks_exact(head_size)
                .skip(h)
                .step_by(n_heads);

            for t in 0..=pos {
                let k = head_k_all_pos.next().unwrap();
                let score = k
                    .iter()
                    .zip(q.iter())
                    .fold(0f32, |acc, (_k, _q)| acc + _k * _q);
                let score = score / (head_size as f32).sqrt();
                unsafe {
                    *att.get_unchecked_mut(t) = score;
                }
            }

            let seq_cached_vals = Self::unchecked_slice(layer_cached_vals, seq_len * kv_dim, layer)
                .chunks_exact(head_size)
                .skip(h)
                .step_by(n_heads);
            operator::softmax(&mut att[..=pos]);

            let dst = xb;
            dst.iter_mut().for_each(|v| *v = 0f32);
            for (vals, att_w) in seq_cached_vals.zip(att.iter()).take(pos + 1) {
                // aggretate timestamp to xb
                for (val, dst) in vals.iter().zip(dst.iter_mut()) {
                    *dst += val * att_w;
                }
            }
        })
    }

    fn ffn(
        &self,
        x: &[f32],
        xb: &mut [f32],
        hb: &mut [f32],
        hb2: &mut [f32],
        layer: usize,
        dim: usize,
        hidden_dim: usize,
    ) {
        let w = &self.weights;
        let rms_ffn_w = Self::unchecked_slice(&w.rms_ffn, dim, layer);
        operator::rmsnorm(xb, rms_ffn_w, x);

        let w1 = Self::unchecked_slice(&w.w1, hidden_dim * dim, layer);
        let w2 = Self::unchecked_slice(&w.w2, hidden_dim * dim, layer);
        let w3 = Self::unchecked_slice(&w.w3, hidden_dim * dim, layer);
        operator::matmul(hb, w1, xb, dim, hidden_dim);
        operator::matmul(hb2, w3, xb, dim, hidden_dim);
        // silu
        operator::silu(hb);

        hb.iter_mut()
            .zip(hb2.iter())
            .for_each(|(h1, &h2)| *h1 *= h2);
        operator::matmul(xb, w2, hb, hidden_dim, dim);
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
    fn unchecked_mut_slice<Q>(s: &mut [Q], size: usize, idx: usize) -> &mut [Q] {
        // let ptr = s.as_ptr() as *mut Q;
        let ptr = s.as_mut_ptr();
        unsafe {
            let offset = ptr.add(idx * size);
            std::slice::from_raw_parts_mut(offset, size)
        }
    }
}
