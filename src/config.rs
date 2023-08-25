use std::io::Read;

use crate::error::Result;
use crate::read_i32;

#[derive(Debug)]
#[repr(C)]
pub struct Config {
    dim: i32,
    hidden_dim: i32,
    n_layers: i32,
    n_heads: i32,
    n_kv_heads: i32,
    vocab_size: i32,
    seq_len: i32,
}

impl Config {
    pub fn from_reader<R: Read>(r: &mut R) -> Result<Self> {
        Ok(Self {
            dim: read_i32(r)?,
            hidden_dim: read_i32(r)?,
            n_layers: read_i32(r)?,
            n_heads: read_i32(r)?,
            n_kv_heads: read_i32(r)?,
            vocab_size: read_i32(r)?,
            seq_len: read_i32(r)?,
        })
    }
}
