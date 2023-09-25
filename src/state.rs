use crate::{config::Config, tensor::DTensor};

pub struct State {
    /// activation at current time stamp
    /// (dim,)
    pub x: DTensor,
    /// same, but inside a residual branch
    /// (dim,)
    pub xb: DTensor,
    /// an additional buffer just for convenience
    /// (dim,)
    pub xb2: DTensor,
    /// buffer for hidden dimension in the ffn
    /// (hidden_dim,)
    pub hb: DTensor,
    /// buffer for hidden dimension in the ffn
    /// (hidden_dim,)
    pub hb2: DTensor,
    /// query
    /// (dim,)
    pub q: DTensor,
    /// key
    /// (kv_dim,)
    pub k: DTensor,
    /// value
    /// (kv_dim,)
    pub v: DTensor,
    /// buffer for scores/attention values
    /// (n_heads, seq_len)
    pub att: DTensor,
    /// output logits
    /// (vocab_size,)
    pub logits: DTensor,
    /// kv cache
    /// (layer, seq_len, kv_dim)
    pub key_cache: DTensor,
    pub value_cache: DTensor,
}

impl State {
    pub fn new(conf: &Config) -> Self {
        let kv_dim = (conf.dim * conf.n_kv_heads) / conf.n_heads;
        Self {
            x: DTensor::new(&[conf.dim]),
            xb: DTensor::new(&[conf.dim]),
            xb2: DTensor::new(&[conf.dim]),
            hb: DTensor::new(&[conf.hidden_dim]),
            hb2: DTensor::new(&[conf.hidden_dim]),
            q: DTensor::new(&[conf.dim]),
            k: DTensor::new(&[kv_dim]),
            v: DTensor::new(&[kv_dim]),
            att: DTensor::new(&[conf.n_heads, conf.seq_len]),
            logits: DTensor::new(&[conf.vocab_size]),
            key_cache: DTensor::new(&[conf.n_layers, conf.seq_len, kv_dim]),
            value_cache: DTensor::new(&[conf.n_layers, conf.seq_len, kv_dim]),
        }
    }
}
