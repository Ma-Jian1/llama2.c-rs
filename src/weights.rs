use std::io::Read;

use crate::{
    config::Config,
    tensor::{DTensor, QTensor},
    Result,
};

#[derive(Debug)]
pub struct Weights {
    /// (vocab_size, dim) == (1, vacab_size, dim)
    pub token_embedding: DTensor,
    /// (layer, dim) rmsnorm weights
    pub rms_att: DTensor,
    /// weights for matmuls. note dim == n_heads * head_size
    /// (layer, dim, n_heads * head_size)
    pub wq: QTensor,
    /// (layer, dim, n_kv_head * head_size)
    pub wk: QTensor,
    /// (layer, dim, n_kv_head * head_size)
    pub wv: QTensor,
    /// (layer, n_heads * head_size, dim)
    pub wo: QTensor,
    /// weights for ffn
    /// (layer, dim)
    pub rms_ffn: DTensor,
    /// (layer, hidden_dim, dim)
    pub w1: QTensor,
    /// (layer, dim, hidden_dim)
    pub w2: QTensor,
    /// (layer, hidden_dim, dim)
    pub w3: QTensor,
    /// final rmsnorm
    /// (dim,)
    pub rms_final: DTensor,
    /// (optional) classifier weights for the logits, on the last layer
    /// (vocab_size, dim)
    pub wcls: Option<DTensor>,
}

impl Weights {
    pub fn from_reader<R: Read>(r: &mut R, cfg: &Config) -> Result<Self> {
        let head_size = cfg.dim / cfg.n_heads;
        let mut weights = Self {
            token_embedding: DTensor::from_reader(r, 1, cfg.vocab_size * cfg.dim)?,
            rms_att: DTensor::from_reader(r, cfg.n_layers, cfg.dim)?,
            wq: QTensor::from_reader(r, cfg.n_layers, cfg.dim * (cfg.n_heads * head_size))?,
            wk: QTensor::from_reader(r, cfg.n_layers, cfg.dim * cfg.n_kv_heads * head_size)?,
            wv: QTensor::from_reader(r, cfg.n_layers, cfg.dim * cfg.n_kv_heads * head_size)?,
            wo: QTensor::from_reader(r, cfg.n_layers, (cfg.n_heads * head_size) * cfg.dim)?,
            rms_ffn: DTensor::from_reader(r, cfg.n_layers, cfg.dim)?,
            w1: QTensor::from_reader(r, cfg.n_layers, cfg.dim * cfg.hidden_dim)?,
            w2: QTensor::from_reader(r, cfg.n_layers, cfg.hidden_dim * cfg.dim)?,
            w3: QTensor::from_reader(r, cfg.n_layers, cfg.dim * cfg.hidden_dim)?,
            rms_final: DTensor::from_reader(r, 1, cfg.dim)?,
            wcls: None,
        };
        if !cfg.shared_weights {
            // skip what used to be freq_cis_real (for RoPE)
            let _ = QTensor::from_reader(r, cfg.seq_len, head_size / 2);
            // skip what used to be freq_cis_imag (for RoPE)
            let _ = QTensor::from_reader(r, cfg.seq_len, head_size / 2);
            weights.wcls = Some(DTensor::from_reader(r, 1, cfg.vocab_size * cfg.dim)?);
        }

        Ok(weights)
    }
}
