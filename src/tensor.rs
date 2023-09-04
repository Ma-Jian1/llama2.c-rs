use std::{io::Read, ops::Deref};

use crate::Result;

pub(crate) const Q_GROUP_SIZE: usize = 32;

#[derive(Debug)]
pub struct Tensor<DType, QType> {
    weight: Vec<QType>,
    // from outer to inner: ..., row, col, group
    layout: Vec<usize>,
    scale: Vec<DType>,
}

impl<DType, QType> Deref for Tensor<DType, QType> {
    type Target = [QType];

    fn deref(&self) -> &Self::Target {
        &self.weight
    }
}

pub type DTensor = Tensor<f32, f32>;
#[cfg(not(feature = "q8"))]
pub type QTensor = DTensor;
#[cfg(feature = "q8")]
pub type QTensor = Tensor<f32, i8>;

// impl<DType, QType> Tensor<DType, QType> {
impl Tensor<f32, f32> {
    pub fn from_reader<R: Read>(r: &mut R, rows: usize, cols: usize) -> Result<Self> {
        debug_assert_eq!(cols % Q_GROUP_SIZE, 0);

        let num = rows * cols;

        let bytes_to_read = num * std::mem::size_of::<f32>();
        let mut raw_tensor = vec![0; bytes_to_read];
        r.read_exact(&mut raw_tensor)?;

        unsafe {
            let float_ptr = raw_tensor.as_ptr() as *const f32;
            let data = std::slice::from_raw_parts(float_ptr, num).to_vec();
            Ok(Self {
                weight: data,
                layout: vec![rows, cols],
                scale: vec![1.0_f32; (rows * cols) / Q_GROUP_SIZE],
            })
        }
    }
}

impl Tensor<f32, i8> {
    pub fn from_reader<R: Read>(r: &mut R, rows: usize, cols: usize) -> Result<Self> {
        debug_assert_eq!(cols % Q_GROUP_SIZE, 0);

        let num = rows * cols;

        let bytes_to_read = num * std::mem::size_of::<f32>();
        let mut raw_tensor = vec![0; bytes_to_read];
        r.read_exact(&mut raw_tensor)?;

        let data = unsafe {
            let float_ptr = raw_tensor.as_ptr() as *const f32;
            std::slice::from_raw_parts(float_ptr, num)
        };

        let (data, scale): (Vec<_>, Vec<_>) = data
            .chunks_exact(Q_GROUP_SIZE)
            .map(|group| {
                let max_val = group.iter().fold(f32::NAN, |acc, &v| v.max(acc));
                let inv_scale = i8::MAX as f32 / max_val;
                let group = group
                    .iter()
                    .map(|&v| (v * inv_scale).round() as i8)
                    .collect::<Vec<_>>();
                (group, 1_f32 / inv_scale)
            })
            .unzip();

        Ok(Self {
            weight: data.into_iter().flatten().collect(),
            layout: vec![rows, cols],
            scale,
        })
    }
}

pub trait AsSlice {
    type Q;
    fn as_slice(&self, idx: usize) -> (&[Self::Q], &[f32]);
}

impl<QType> AsSlice for Tensor<f32, QType> {
    type Q = QType;
    fn as_slice(&self, idx: usize) -> (&[Self::Q], &[f32]) {
        let (rows, cols) = (self.layout[0], self.layout[1]);
        debug_assert!(idx < rows);

        let scale_size = cols / Q_GROUP_SIZE;
        let ptr = self.weight.as_ptr();
        let scale_ptr = self.scale.as_ptr();
        unsafe {
            let offset = ptr.add(idx * cols);
            let scale_offset = scale_ptr.add(idx * scale_size);
            (
                std::slice::from_raw_parts(offset, cols),
                std::slice::from_raw_parts(scale_offset, scale_size),
            )
        }
    }
}
