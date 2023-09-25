use std::{
    io::Read,
    ops::{Deref, DerefMut, Index, IndexMut},
};

use crate::Result;

// in case we will support f16/bf16
pub type Float = f32;

#[cfg(feature = "q8")]
pub(crate) const Q_GROUP_SIZE: usize = 32;

pub type DTensor = Tensor<Float>;
#[cfg(not(feature = "q8"))]
pub type QTensor = DTensor;
#[cfg(feature = "q8")]
pub type QTensor = Tensor<i8>;

#[derive(Debug)]
pub struct Tensor<QType> {
    value: Vec<QType>,
    // row-wise: ..., layer, row, col
    shape: Vec<usize>,
    numel: usize,
    // for group which col % group = 0
    // ie, its scale: ..., layer, col/group
    #[cfg(feature = "q8")]
    scale: Vec<Float>,
}

impl<QType> Tensor<QType> {
    pub fn new(shape: &[usize]) -> Self
    where
        QType: Clone + Default,
    {
        assert!(!shape.is_empty());
        #[cfg(feature = "q8")]
        assert_eq!(shape.last().unwrap() % Q_GROUP_SIZE, 0);

        let numel = shape.iter().product();
        Self {
            value: vec![QType::default(); numel],
            shape: shape.to_owned(),
            numel,
            #[cfg(feature = "q8")]
            scale: vec![Float::default(); numel / Q_GROUP_SIZE],
        }
    }

    pub fn copy_from_slice(&mut self, src: &[QType])
    where
        QType: Copy,
    {
        assert_eq!(self.value.len(), src.len());
        self.value.as_mut_slice().copy_from_slice(src);
    }
}

impl<QType> Deref for Tensor<QType> {
    type Target = [QType];

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<QType> DerefMut for Tensor<QType> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

// Tensor<Float> does not use scale
impl Index<usize> for Tensor<Float> {
    type Output = [Float];

    fn index(&self, index: usize) -> &Self::Output {
        let douter = self.shape[0];
        assert!(index < douter);
        let numel = self.numel / douter;

        let ptr = self.value.as_ptr();
        unsafe {
            let offset = ptr.add(index * numel);
            std::slice::from_raw_parts(offset, numel)
        }
    }
}

impl IndexMut<usize> for Tensor<Float> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let douter = self.shape[0];
        assert!(index < douter);
        let numel = self.numel / douter;

        let ptr = self.value.as_mut_ptr();
        unsafe {
            let offset = ptr.add(index * numel);
            std::slice::from_raw_parts_mut(offset, numel)
        }
    }
}

impl Index<(usize, usize)> for Tensor<Float> {
    type Output = [Float];

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        assert!(self.shape.len() > 2);
        let (douter, dinner) = (self.shape[0], self.shape[1]);
        assert!(index.0 < douter);
        assert!(index.1 < dinner);
        let numel = self.numel / (douter * dinner);

        let ptr = self.value.as_ptr();
        unsafe {
            let offset = ptr.add(index.0 * (self.numel / douter) + index.1 * numel);
            std::slice::from_raw_parts(offset, numel)
        }
    }
}

impl IndexMut<(usize, usize)> for Tensor<Float> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        assert!(self.shape.len() > 2);
        let (douter, dinner) = (self.shape[0], self.shape[1]);
        assert!(index.0 < douter);
        assert!(index.1 < dinner);
        let numel = self.numel / (douter * dinner);

        let ptr = self.value.as_mut_ptr();
        unsafe {
            let offset = ptr.add(index.0 * (self.numel / douter) + index.1 * numel);
            std::slice::from_raw_parts_mut(offset, numel)
        }
    }
}

/// Tensor<QType> which use scale
#[cfg(feature = "q8")]
impl<QType> Tensor<QType> {
    fn index_row_and_scale(&self, idx: usize) -> (&[QType], &[Float]) {
        let (rows, cols) = (self.shape[0], self.shape[1]);
        assert!(idx < rows);
        assert_eq!(cols % Q_GROUP_SIZE, 0);

        let scale_size = cols / Q_GROUP_SIZE;
        let ptr = self.value.as_ptr();
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

#[cfg(feature = "q8")]
impl<QType> Tensor<QType>
where
    QType: Into<Float> + Copy,
{
    pub fn rmsnorm(&self, out: &mut [Float], x: &[Float], row: usize) {
        assert_eq!(out.len(), x.len());

        let (w, ws) = self.index_row_and_scale(row);
        assert_eq!(w.len(), x.len());

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
    pub fn matmul(&self, out: &mut [Float], x: &[Float], n: usize, d: usize, row: usize) {
        assert_eq!(out.len(), d);
        assert_eq!(x.len(), n);
        assert_eq!(n % Q_GROUP_SIZE, 0);

        let (w, ws) = self.index_row_and_scale(row);
        assert_eq!(w.len(), d * n);

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
                .map(Into::<Float>::into)
                .zip(ws)
                .zip(x.iter())
                .fold(0 as Float, |acc, ((w, ws), x)| acc + ws * w * x);
        }
    }
}

impl Tensor<Float> {
    pub fn from_reader<R: Read>(r: &mut R, rows: usize, cols: usize) -> Result<Self> {
        #[cfg(feature = "q8")]
        assert_eq!(cols % Q_GROUP_SIZE, 0);

        let num = rows * cols;

        let bytes_to_read = num * std::mem::size_of::<Float>();
        let mut raw_tensor = vec![0; bytes_to_read];
        r.read_exact(&mut raw_tensor)?;

        unsafe {
            let float_ptr = raw_tensor.as_ptr() as *const Float;
            let data = std::slice::from_raw_parts(float_ptr, num).to_vec();
            Ok(Self {
                value: data,
                shape: vec![rows, cols],
                numel: rows * cols,
                #[cfg(feature = "q8")]
                scale: vec![1 as Float; (rows * cols) / Q_GROUP_SIZE],
            })
        }
    }

    /// Treat s as 2-dimension array: [[Float; size]; x] and return &s[idx][..]
    pub fn unchecked_slice(&self, size: usize, idx: usize) -> &[Float] {
        let ptr = self.value.as_ptr();
        unsafe {
            let offset = ptr.add(idx * size);
            std::slice::from_raw_parts(offset, size)
        }
    }

    /// Treat s as 2-dimension array: [[Float; size]; x] and return &s[idx][..]
    pub fn unchecked_mut_slice(&mut self, size: usize, idx: usize) -> &mut [Float] {
        let ptr = self.value.as_mut_ptr();
        unsafe {
            let offset = ptr.add(idx * size);
            std::slice::from_raw_parts_mut(offset, size)
        }
    }
}

#[cfg(not(feature = "q8"))]
impl Tensor<Float> {
    pub fn rmsnorm(&self, out: &mut [Float], x: &[Float], row: usize) {
        assert_eq!(out.len(), x.len());

        let w = &self[row];
        assert_eq!(w.len(), x.len());

        // sum(x^2)
        let ss = x.iter().fold(0 as Float, |init, &v| init + v * v) / (x.len() as Float);
        // 1.0 / sqrt(sum(x^2) + 1e-5)
        let ss = 1.0 / (ss + 1e-5).sqrt();
        out.iter_mut()
            .zip(w.iter().zip(x.iter()))
            .for_each(|(o, (w, x))| *o = w * (ss * x));
    }

    /// W(d, n) * x(n,) -> out(d,)
    pub fn matmul(&self, out: &mut [Float], x: &[Float], n: usize, d: usize, row: usize) {
        assert_eq!(out.len(), d);
        assert_eq!(x.len(), n);

        let w = &self[row];
        assert_eq!(w.len(), d * n);

        for (row, out) in w.chunks_exact(n).zip(out.iter_mut()) {
            *out = row
                .iter()
                .zip(x.iter())
                .fold(0 as Float, |acc, (w, x)| acc + w * x);
        }
    }
}

#[cfg(feature = "q8")]
impl Tensor<i8> {
    pub fn from_reader<R: Read>(r: &mut R, rows: usize, cols: usize) -> Result<Self> {
        assert_eq!(cols % Q_GROUP_SIZE, 0);

        let num = rows * cols;

        let bytes_to_read = num * std::mem::size_of::<Float>();
        let mut raw_tensor = vec![0; bytes_to_read];
        r.read_exact(&mut raw_tensor)?;

        let data = unsafe {
            let float_ptr = raw_tensor.as_ptr() as *const Float;
            std::slice::from_raw_parts(float_ptr, num)
        };

        let (data, scale): (Vec<_>, Vec<_>) = data
            .chunks_exact(Q_GROUP_SIZE)
            .map(|group| {
                let max_val = group.iter().fold(Float::NAN, |acc, &v| v.max(acc));
                let inv_scale = i8::MAX as Float / max_val;
                let group = group
                    .iter()
                    .map(|&v| (v * inv_scale).round() as i8)
                    .collect::<Vec<_>>();
                (group, 1_f32 / inv_scale)
            })
            .unzip();

        Ok(Self {
            value: data.into_iter().flatten().collect(),
            shape: vec![rows, cols],
            numel: rows * cols,
            scale,
        })
    }
}
