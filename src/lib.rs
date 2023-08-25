use std::io::Read;

use error::Result;

pub mod config;
pub mod error;

pub(crate) fn read_i32<R: Read>(r: &mut R) -> Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}
