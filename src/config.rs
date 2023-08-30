use std::io::Read;

use crate::Result;

#[derive(Debug)]
#[repr(C)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub shared_weights: bool,
}

impl Config {
    pub fn from_reader<R: Read>(r: &mut R) -> Result<Self> {
        const CONF_VALS: usize = 7;
        const CONF_SIZE: usize = std::mem::size_of::<[i32; CONF_VALS]>();
        let mut buf = [0u8; CONF_SIZE];
        r.read_exact(&mut buf)?;
        let raw_conf = unsafe { std::mem::transmute::<[u8; CONF_SIZE], [i32; CONF_VALS]>(buf) };
        let (vocab_size, shared_weights) = if raw_conf[5] > 0 {
            (raw_conf[5] as usize, true)
        } else {
            (-raw_conf[5] as usize, false)
        };
        Ok(Self {
            dim: raw_conf[0] as usize,
            hidden_dim: raw_conf[1] as usize,
            n_layers: raw_conf[2] as usize,
            n_heads: raw_conf[3] as usize,
            n_kv_heads: raw_conf[4] as usize,
            vocab_size,
            seq_len: raw_conf[6] as usize,
            shared_weights,
        })
    }
}
