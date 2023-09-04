use std::{
    io::Read,
    ops::{Deref, Index},
};

use crate::{Float, Result};

pub(crate) const Q_GROUP_SIZE: usize = 32;

pub type DTensor = Tensor<Float>;
#[cfg(not(feature = "q8"))]
pub type QTensor = DTensor;
#[cfg(feature = "q8")]
pub type QTensor = Tensor<i8>;

#[derive(Debug)]
pub struct Tensor<QType> {
    weight: Vec<QType>,
    // from outer to inner: ..., row, col(group)
    layout: Vec<usize>,
    scale: Vec<Float>,
}

impl<QType> Deref for Tensor<QType> {
    type Target = [QType];

    fn deref(&self) -> &Self::Target {
        &self.weight
    }
}

/// Tensor<f32> does not use scale
impl Index<usize> for Tensor<f32> {
    type Output = [f32];

    fn index(&self, index: usize) -> &Self::Output {
        let (rows, cols) = (self.layout[0], self.layout[1]);
        debug_assert!(index < rows);

        let ptr = self.weight.as_ptr();
        unsafe {
            let offset = ptr.add(index * cols);
            std::slice::from_raw_parts(offset, cols)
        }
    }
}

/// Tensor<QType> which use scale
impl<QType> Tensor<QType> {
    fn index_row_and_scale(&self, idx: usize) -> (&[QType], &[f32]) {
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

impl<QType> Tensor<QType>
where
    QType: Into<Float> + Copy,
{
    pub fn rmsnorm(&self, out: &mut [Float], x: &[Float], row: usize) {
        debug_assert_eq!(out.len(), x.len());

        let (w, ws) = self.index_row_and_scale(row);
        debug_assert_eq!(w.len(), x.len());

        let ws = ws
            .iter()
            .flat_map(|ws| std::iter::repeat(ws).take(Q_GROUP_SIZE));
        let w = w.iter().copied().map(Into::<Float>::into).zip(ws);

        // sum(x^2)
        let ss = x.iter().fold(0 as Float, |init, &v| init + v * v) / (x.len() as Float);
        // 1.0 / sqrt(sum(x^2) + 1e-5)
        let ss = 1.0 / (ss + 1e-5).sqrt();
        out.iter_mut()
            .zip(w.zip(x.iter()))
            .for_each(|(o, ((w, ws), x))| *o = *ws * w * (ss * x));
    }

    /// W(d, n) * x(n,) -> out(d,)
    pub fn matmul_3d(&self, out: &mut [f32], x: &[f32], n: usize, d: usize, row: usize) {
        debug_assert_eq!(out.len(), d);
        debug_assert_eq!(x.len(), n);
        debug_assert_eq!(n % Q_GROUP_SIZE, 0);

        let (w, ws) = self.index_row_and_scale(row);
        debug_assert_eq!(w.len(), d * n);

        for ((row, ws), out) in w
            .chunks_exact(n)
            .zip(ws.chunks_exact(n / Q_GROUP_SIZE))
            .zip(out.iter_mut())
        {
            let ws = ws
                .iter()
                .flat_map(|ws| std::iter::repeat(ws).take(Q_GROUP_SIZE));
            *out = row
                .iter()
                .copied()
                .map(Into::<f32>::into)
                .zip(ws)
                .zip(x.iter())
                .fold(0f32, |acc, ((w, ws), x)| acc + ws * w * x);
        }
    }

    /// W(d, n) * x(n,) -> out(d,)
    pub fn matmul(&self, out: &mut [f32], x: &[f32], n: usize, d: usize)
    where
        QType: Into<f32> + Copy,
    {
        let w = &self.weight;

        debug_assert_eq!(w.len(), d * n);
        debug_assert_eq!(out.len(), d);
        debug_assert_eq!(x.len(), n);

        for (row, o) in w.chunks_exact(n).zip(out.iter_mut()) {
            *o = row
                .iter()
                .zip(x.iter())
                .fold(0f32, |acc, (&w, &x)| acc + w.into() * x);
        }
    }
}

impl Tensor<f32> {
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

impl Tensor<i8> {
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
