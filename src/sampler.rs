use crate::operator;

pub struct Sampler {
    temperature: f32,
}

impl Sampler {
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }

    pub fn sample(&self, logits: &mut [f32]) -> usize {
        if self.temperature == 0.0 {
            operator::argmax(logits)
        } else {
            logits
                .iter_mut()
                .for_each(|logit| *logit /= self.temperature);
            operator::softmax(logits);
            operator::sample(logits)
        }
    }
}
