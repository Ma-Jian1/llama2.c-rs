use crate::kernel;

pub struct Sampler {
    temperature: f32,
}

impl Sampler {
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }

    pub fn sample(&self, logits: &mut [f32]) -> usize {
        if self.temperature == 0.0 {
            kernel::argmax(logits)
        } else {
            logits
                .iter_mut()
                .for_each(|logit| *logit /= self.temperature);
            kernel::softmax(logits);
            kernel::sample(logits)
        }
    }
}
