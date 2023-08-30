use std::io::Read;

use crate::{config::Config, read_tensor, Result};

type DType = f32;

#[derive(Debug)]
pub struct TransformerWeights {
    /// (vocab_size, dim)
    pub token_embedding_table: Vec<DType>,
    /// (layer, dim) rmsnorm weights
    pub rms_att_weight: Vec<DType>,
    /// (layer, dim)
    pub rms_ffn_weight: Vec<DType>,
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
    /// (layer, hidden_dim, dim)
    pub w1: Vec<DType>,
    /// (layer, dim, hidden_dim)
    pub w2: Vec<DType>,
    /// (layer, hidden_dim, dim)
    pub w3: Vec<DType>,
    /// final rmsnorm
    /// (dim,)
    pub rms_final_weight: Vec<DType>,
    /// (optional) classifier weights for the logits, on the last layer
    /// (vocab_size, dim)
    pub wcls: Option<Vec<DType>>,
}

impl TransformerWeights {
    pub fn from_reader<R: Read>(r: &mut R, conf: &Config) -> Result<Self> {
        let head_size = conf.dim / conf.n_heads;
        let mut weights = Self {
            token_embedding_table: read_tensor(r, conf.vocab_size * conf.dim)?,
            rms_att_weight: read_tensor(r, conf.n_layers * conf.dim)?,
            rms_ffn_weight: read_tensor(r, conf.n_layers * conf.dim)?,
            wq: read_tensor(r, conf.n_layers * conf.dim * conf.dim)?,
            wk: read_tensor(r, conf.n_layers * conf.dim * conf.n_kv_heads * head_size)?,
            wv: read_tensor(r, conf.n_layers * conf.dim * conf.n_kv_heads * head_size)?,
            wo: read_tensor(r, conf.n_layers * conf.dim * conf.dim)?,
            w1: read_tensor(r, conf.n_layers * conf.hidden_dim * conf.dim)?,
            w2: read_tensor(r, conf.n_layers * conf.dim * conf.hidden_dim)?,
            w3: read_tensor(r, conf.n_layers * conf.hidden_dim * conf.dim)?,
            rms_final_weight: read_tensor(r, conf.dim)?,
            wcls: None,
        };
        if !conf.shared_weights {
            weights.wcls = Some(read_tensor(r, conf.vocab_size * conf.dim)?);
        }

        Ok(weights)
    }
}
