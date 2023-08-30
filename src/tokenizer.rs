use std::fs::File;
use std::io::Read;

use crate::Result;

pub struct Tokenizer {
    vocab_size: usize,
    max_token_length: u32,
    vocab_scores: Vec<f32>,
    bytes: Vec<u8>,
    offsets: Vec<usize>,
}

impl Tokenizer {
    pub fn new(tokenizer_path: &str, vocab_size: usize) -> Result<Self> {
        let mut file = File::open(tokenizer_path)?;

        let max_token_length = read_i32(&mut file)? as u32;

        let mut vocab_scores = Vec::<f32>::new();
        let mut bytes = Vec::<u8>::new();
        let mut offsets = vec![0; 1];

        let mut val = [0; 1];
        for _i in 0..vocab_size {
            let s = read_f32(&mut file)?;
            vocab_scores.push(s);

            let l = read_i32(&mut file)?;
            offsets.push(offsets.last().unwrap() + l as usize);
            for _ in 0..l {
                file.read_exact(&mut val)?;
                bytes.extend(val);
            }
        }

        assert_eq!(offsets.len(), vocab_size + 1);
        Ok(Self {
            vocab_size,
            max_token_length,
            vocab_scores,
            bytes,
            offsets,
        })
    }

    pub fn get_token(&self, idx: usize) -> Result<&str> {
        let (begin, end) = (self.offsets[idx], self.offsets[idx + 1]);
        let b = &self.bytes[begin..end];
        Ok(std::str::from_utf8(b)?)
    }
}

fn read_i32<R: Read>(r: &mut R) -> Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

// fn read_primitive<T, R: Read>(r: &mut R) -> Result<T> {
//     let mut buf = [0u8; std::mem::size_of::<T>()];
//     r.read_exact(&mut buf)?;
//     Ok(unsafe { std::mem::transmute::<[u8; std::mem::size_of::<T>()], T>(buf) })
// }
