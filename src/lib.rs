use std::io::Read;

pub mod config;
pub mod kernel;
pub mod state;
pub mod tensor;
pub mod weights;

mod error;
mod model;
#[cfg(feature = "q8")]
mod quant;
mod sampler;
mod tokenizer;

pub use error::Llama2Error;
pub use error::Result;
pub use model::Llama2Model;
pub use sampler::Sampler;
pub use tokenizer::{SpecialToken, Tokenizer};

// in case we will support f16/bf16
pub type Float = f32;

// fn read_primitive<T, R: Read>(r: &mut R) -> Result<T> {
//     let mut buf = [0u8; std::mem::size_of::<T>()];
//     r.read_exact(&mut buf)?;
//     Ok(unsafe { std::mem::transmute::<[u8; std::mem::size_of::<T>()], T>(buf) })
// }

pub(crate) fn read_i32<R: Read>(r: &mut R) -> Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

pub(crate) fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}
