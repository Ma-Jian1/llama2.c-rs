use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;

use crate::tensor::Float;

pub fn argmax(x: &[Float]) -> usize {
    assert!(!x.is_empty());
    x.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .expect("argmax")
}

pub fn softmax(x: &mut [Float]) {
    assert!(!x.is_empty());
    let max_val = x.iter().fold(Float::NAN, |acc, &v| v.max(acc));
    let mut sum = 0 as Float;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    x.iter_mut().for_each(|v| *v /= sum);
}

pub fn rope(x: &mut [Float], head_size: usize, pos: usize) {
    // q may not equal to k, but they all should be divided by head_size
    for head in x.chunks_exact_mut(head_size) {
        for (i, v) in head.chunks_mut(2).enumerate() {
            let freq = 1 as Float / 10000f32.powf(2 as Float * i as Float / head_size as Float);
            let val = pos as Float * freq;
            let fcr = val.cos();
            let fci = val.sin();

            let v0 = v[0];
            let v1 = v[1];
            v[0] = v0 * fcr - v1 * fci;
            v[1] = v0 * fci + v1 * fcr;
        }
    }
}

/// silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
pub fn silu(x: &mut [Float]) {
    x.iter_mut()
        .for_each(|v| *v = (*v) * (1 as Float / (1 as Float + (-*v).exp())));
}

pub fn sample(probs: &[Float]) -> usize {
    let mut rng = SmallRng::from_entropy();
    let r = rng.gen::<Float>();

    let mut cdf = 0 as Float;
    for (idx, p) in probs.iter().enumerate() {
        cdf += *p;
        if r < cdf {
            return idx;
        }
    }
    probs.len() - 1
}
