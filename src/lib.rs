use std::io::Read;

use error::Result;

pub mod config;
pub mod error;
pub mod state;
pub mod weights;

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
