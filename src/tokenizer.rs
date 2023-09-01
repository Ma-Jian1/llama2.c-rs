use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

use crate::{read_f32, read_i32, Result};

#[allow(dead_code)]
pub enum SpecialToken {
    Unk = 0,
    Bos = 1,
    Eos = 2,
}

pub struct Tokenizer {
    vocab_size: usize,
    max_token_length: i32,
    vocab_scores: Vec<f32>,
    bytes: Vec<u8>,
    offsets: Vec<usize>,
    ascii_pieces: Vec<String>,
    word_to_token_id: HashMap<String, usize>,
}

impl Tokenizer {
    pub fn new(tokenizer_path: &str, vocab_size: usize) -> Result<Self> {
        let mut file = File::open(tokenizer_path)?;

        let max_token_length = read_i32(&mut file)?;

        let mut vocab_scores = Vec::<f32>::new();
        let mut bytes = Vec::<u8>::new();
        let mut offsets = vec![0; 1];
        let mut word_to_token_id = HashMap::new();

        let mut val = [0; 1];
        for i in 0..vocab_size {
            let s = read_f32(&mut file)?;
            vocab_scores.push(s);

            let len = read_i32(&mut file)?;

            offsets.push(offsets.last().unwrap() + len as usize);
            for _ in 0..len {
                file.read_exact(&mut val)?;
                bytes.extend(val);
            }

            let (begin, end) = (offsets[i], offsets[i + 1]);
            let word = std::str::from_utf8(&bytes[begin..end])?.to_string();
            word_to_token_id.insert(word, i);
        }
        debug_assert_eq!(offsets.len(), vocab_size + 1);

        let ascii_pieces: Vec<String> = (0..=256).map(|i| (i as u8 as char).to_string()).collect();

        Ok(Self {
            vocab_size,
            max_token_length,
            vocab_scores,
            bytes,
            offsets,
            ascii_pieces,
            word_to_token_id,
        })
    }

    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Result<Vec<usize>> {
        let mut tokens: Vec<usize> = Vec::new();
        if bos {
            tokens.push(SpecialToken::Bos as usize);
        }
        if !text.is_empty() {
            let dummy_prefix = self.word_to_token_id.get(" ").unwrap();
            tokens.push(*dummy_prefix);
        }

        for ch in text.chars() {
            let ch_str = ch.to_string();
            match self.word_to_token_id.get(&ch_str) {
                Some(&id) => tokens.push(id),
                None => {
                    // byte_fallback encoding: just encode each byte as a token
                    // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
                    // so the individual bytes only start at index 3
                    for byte in ch_str.as_bytes() {
                        tokens.push(*byte as usize + 3);
                    }
                }
            }
        }

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_id = 0;
            let mut best_idx = None;

            for i in 0..(tokens.len() - 1) {
                let pair = format!(
                    "{}{}",
                    self.get_token(tokens[i])?,
                    self.get_token(tokens[i + 1])?
                );
                if let Some(&id) = self.word_to_token_id.get(&pair) {
                    if self.vocab_scores[id] > best_score {
                        best_score = self.vocab_scores[id];
                        best_id = id;
                        best_idx = Some(i);
                    }
                }
            }

            if let Some(idx) = best_idx {
                tokens[idx] = best_id;
                tokens.remove(idx + 1);
            } else {
                break;
            }
        }

        if eos {
            tokens.push(SpecialToken::Eos as usize);
        }

        Ok(tokens)
    }

    pub fn decode(&self, idx: usize, ltrim: bool) -> Result<&str> {
        let mut piece = self.get_token(idx)?;
        if ltrim {
            piece = piece.strip_prefix(' ').unwrap_or(piece);
        }
        if let Some(hex) = piece.strip_prefix("<0x") {
            if let Ok(byte) = usize::from_str_radix(&hex[..2], 16) {
                return Ok(&self.ascii_pieces[byte]);
            }
        }
        Ok(piece)
    }

    fn get_token(&self, idx: usize) -> Result<&str> {
        let (begin, end) = (self.offsets[idx], self.offsets[idx + 1]);
        Ok(std::str::from_utf8(&self.bytes[begin..end])?)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn decode_token() {
        let tokenizer = Tokenizer::new("llama2.c/tokenizer.bin", 32000).unwrap();
        assert_eq!(
            tokenizer.decode(SpecialToken::Unk as usize, false).unwrap(),
            "<unk>"
        );
        assert_eq!(
            tokenizer.decode(SpecialToken::Bos as usize, false).unwrap(),
            "\n<s>\n"
        );
        assert_eq!(
            tokenizer.decode(SpecialToken::Eos as usize, false).unwrap(),
            "\n</s>\n"
        );
        // test for (<0x00> .. <0xFF>) in tokenizer.bin, which offset start from 3
        for idx in 0..256 {
            assert_eq!(
                tokenizer.decode(idx + 3, false).unwrap(),
                (idx as u8 as char).to_string()
            );
        }
    }
}
