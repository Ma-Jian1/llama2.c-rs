use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;

pub fn rmsnorm(out: &mut [f32], w: &[f32], x: &[f32]) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(w.len(), x.len());

    // sum(x^2)
    let ss = x.iter().fold(0f32, |init, &v| init + v * v) / (x.len() as f32);
    // 1.0 / sqrt(sum(x^2) + 1e-5)
    let ss = 1.0 / (ss + 1e-5).sqrt();
    out.iter_mut()
        .zip(w.iter().zip(x.iter()))
        .for_each(|(o, (w, x))| *o = w * x * ss);
}

pub fn rmsnorm_inplace(x: &mut [f32], w: &[f32]) {
    debug_assert_eq!(w.len(), x.len());

    // sum(x^2)
    let ss = x.iter().fold(0f32, |init, &v| init + v * v) / (x.len() as f32);
    // 1.0 / sqrt(sum(x^2) + 1e-5)
    let ss = 1.0 / (ss + 1e-5).sqrt();
    x.iter_mut()
        .zip(w.iter())
        .for_each(|(x, w)| *x = w * *x * ss);
}

/// W(d, n) * x(n,) -> xout(d)
pub fn matmul(out: &mut [f32], w: &[f32], x: &[f32], n: usize, d: usize) {
    debug_assert_eq!(w.len(), d * n);
    debug_assert_eq!(out.len(), d);
    debug_assert_eq!(x.len(), n);

    for (row, out_elem) in w.chunks_exact(n).zip(out.iter_mut()) {
        let val = row
            .iter()
            .zip(x.iter())
            .fold(0f32, |acc, (&w, &x)| acc + w * x);
        *out_elem = val;
    }
}

pub fn argmax(x: &[f32]) -> usize {
    debug_assert!(!x.is_empty());
    x.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap()
}

pub fn softmax(x: &mut [f32]) {
    debug_assert!(!x.is_empty());
    let max_val = x.iter().fold(f32::NAN, |acc, &v| v.max(acc));
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    x.iter_mut().for_each(|v| *v /= sum);
}

pub fn rope(x: &mut [f32], head_size: usize, pos: usize) {
    // q may not equal to k, but they all should be divided by head_size
    for head in x.chunks_exact_mut(head_size) {
        for (i, v) in head.chunks_mut(2).enumerate() {
            let freq = 1.0 / 10000f32.powf(i as f32 / head_size as f32);
            let val = pos as f32 * freq;
            let fcr = val.cos();
            let fci = val.sin();

            let v0 = v[0];
            let v1 = v[1];
            v[0] = v0 * fcr - v1 * fci;
            v[1] = v0 * fcr + v1 * fci;
        }
    }
}

pub fn silu(x: &mut [f32]) {
    x.iter_mut()
        .for_each(|v| *v = (*v) * (1f32 / (1f32 + (-*v).exp())));
}

pub fn sample(probs: &[f32]) -> usize {
    let mut rng = SmallRng::from_entropy();
    let r = rng.gen::<f32>();

    let mut cdf = 0f32;
    for (idx, p) in probs.iter().enumerate() {
        cdf += *p;
        if r < cdf {
            return idx;
        }
    }
    probs.len() - 1
}
