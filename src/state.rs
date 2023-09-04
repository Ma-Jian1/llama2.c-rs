use crate::{config::Config, Float};

pub struct State {
    /// activation at current time stamp
    /// (dim,)
    pub x: Vec<Float>,
    /// same, but inside a residual branch
    /// (dim,)
    pub xb: Vec<Float>,
    /// an additional buffer just for convenience
    /// (dim,)
    pub xb2: Vec<Float>,
    /// buffer for hidden dimension in the ffn
    /// (hidden_dim,)
    pub hb: Vec<Float>,
    /// buffer for hidden dimension in the ffn
    /// (hidden_dim,)
    pub hb2: Vec<Float>,
    /// query
    /// (dim,)
    pub q: Vec<Float>,
    /// key
    /// (kv_dim,)
    pub k: Vec<Float>,
    /// value
    /// (kv_dim,)
    pub v: Vec<Float>,
    /// buffer for scores/attention values
    /// (n_heads, seq_len)
    pub att: Vec<Float>,
    /// output logits
    /// (vocab_size,)
    pub logits: Vec<Float>,
    /// kv cache
    /// (layer, seq_len, kv_dim)
    pub key_cache: Vec<Float>,
    pub value_cache: Vec<Float>,
}

impl State {
    pub fn new(conf: &Config) -> Self {
        let kv_dim = (conf.dim * conf.n_kv_heads) / conf.n_heads;
        Self {
            x: vec![0.0; conf.dim],
            xb: vec![0.0; conf.dim],
            xb2: vec![0.0; conf.dim],
            hb: vec![0.0; conf.hidden_dim],
            hb2: vec![0.0; conf.hidden_dim],
            q: vec![0.0; conf.dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            att: vec![0.0; conf.n_heads * conf.seq_len],
            logits: vec![0.0; conf.vocab_size],
            key_cache: vec![0.0; conf.n_layers * conf.seq_len * kv_dim],
            value_cache: vec![0.0; conf.n_layers * conf.seq_len * kv_dim],
        }
    }
}
