use std::io::Read;

use crate::{config::Config, read_tensor, Result};

type DType = f32;

#[derive(Debug)]
pub struct Weights {
    /// (vocab_size, dim)
    pub token_embedding: Vec<DType>,
    /// (layer, dim) rmsnorm weights
    pub rms_att: Vec<DType>,
    /// weights for matmuls. note dim == n_heads * head_size
    /// (layer, dim, n_heads * head_size)
    pub wq: Vec<DType>,
    /// (layer, dim, n_kv_head * head_size)
    pub wk: Vec<DType>,
    /// (layer, dim, n_kv_head * head_size)
    pub wv: Vec<DType>,
    /// (layer, n_heads * head_size, dim)
    pub wo: Vec<DType>,
    /// weights for ffn
    /// (layer, dim)
    pub rms_ffn: Vec<DType>,
    /// (layer, hidden_dim, dim)
    pub w1: Vec<DType>,
    /// (layer, dim, hidden_dim)
    pub w2: Vec<DType>,
    /// (layer, hidden_dim, dim)
    pub w3: Vec<DType>,
    /// final rmsnorm
    /// (dim,)
    pub rms_final: Vec<DType>,
    /// (optional) classifier weights for the logits, on the last layer
    /// (vocab_size, dim)
    pub wcls: Option<Vec<DType>>,
}

impl Weights {
    pub fn from_reader<R: Read>(r: &mut R, cfg: &Config) -> Result<Self> {
        let head_size = cfg.dim / cfg.n_heads;
        let mut weights = Self {
            token_embedding: read_tensor(r, cfg.vocab_size * cfg.dim)?,
            rms_att: read_tensor(r, cfg.n_layers * cfg.dim)?,
            wq: read_tensor(r, cfg.n_layers * cfg.dim * (cfg.n_heads * head_size))?,
            wk: read_tensor(r, cfg.n_layers * cfg.dim * cfg.n_kv_heads * head_size)?,
            wv: read_tensor(r, cfg.n_layers * cfg.dim * cfg.n_kv_heads * head_size)?,
            wo: read_tensor(r, cfg.n_layers * (cfg.n_heads * head_size) * cfg.dim)?,
            rms_ffn: read_tensor(r, cfg.n_layers * cfg.dim)?,
            w1: read_tensor(r, cfg.n_layers * cfg.dim * cfg.hidden_dim)?,
            w2: read_tensor(r, cfg.n_layers * cfg.hidden_dim * cfg.dim)?,
            w3: read_tensor(r, cfg.n_layers * cfg.dim * cfg.hidden_dim)?,
            rms_final: read_tensor(r, cfg.dim)?,
            wcls: None,
        };
        if !cfg.shared_weights {
            // skip what used to be freq_cis_real (for RoPE)
            let _ = read_tensor(r, cfg.seq_len * head_size / 2);
            // skip what used to be freq_cis_imag (for RoPE)
            let _ = read_tensor(r, cfg.seq_len * head_size / 2);
            weights.wcls = Some(read_tensor(r, cfg.vocab_size * cfg.dim)?);
        }

        Ok(weights)
    }
}
