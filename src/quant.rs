use crate::{tensor::Q_GROUP_SIZE, Float};

pub fn quantize(q: &mut [i8], s: &mut [Float], x: &[Float]) {
    assert_eq!(q.len(), x.len());
    assert_eq!(x.len() % Q_GROUP_SIZE, 0);

    q.chunks_exact_mut(Q_GROUP_SIZE)
        .zip(s.iter_mut())
        .zip(x.chunks_exact(Q_GROUP_SIZE))
        .for_each(|((q_group, s), x_group)| {
            let max_val = x_group.iter().fold(Float::NAN, |acc, &v| v.abs().max(acc));
            let inv_scale = i8::MAX as Float / max_val;
            q_group
                .iter_mut()
                .zip(x.iter())
                .for_each(|(q, x)| *q = (x * inv_scale).round() as i8);
            *s = 1_f32 / inv_scale;
        });
}
