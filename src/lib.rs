use std::io::Read;

pub mod config;
pub mod state;
pub mod tokenizer;
pub mod weights;

mod error;

pub use error::Llama2Error;
pub use error::Result;

// pub(crate) fn read_i32<R: Read>(r: &mut R) -> Result<i32> {
//     let mut buf = [0u8; 4];
//     r.read_exact(&mut buf)?;
//     Ok(i32::from_le_bytes(buf))
// }

pub(crate) fn read_tensor<R: Read>(r: &mut R, num: usize) -> Result<Vec<f32>> {
    let bytes_to_read = num * std::mem::size_of::<f32>();
    let mut raw_tensor = vec![0; bytes_to_read];
    r.read_exact(&mut raw_tensor)?;
    unsafe {
        let float_ptr = raw_tensor.as_ptr() as *const f32;
        let tensor = std::slice::from_raw_parts(float_ptr, num);
        Ok(tensor.to_vec())
    }
}
