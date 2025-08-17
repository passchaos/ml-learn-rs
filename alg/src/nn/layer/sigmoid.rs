use num::Float;

use crate::nn::{Mat, layer::LayerWard};

pub struct Sigmoid {
    out: Option<Mat>,
}

impl LayerWard for Sigmoid {
    fn forward(&mut self, input: &Mat) -> Mat {
        let out = input.mapv(|a| 1.0 / (1.0 + (-a).exp()));

        self.out = Some(out.clone());

        out
    }

    fn backward(&mut self, grad: &Mat) -> Mat {
        let mut dx = grad.clone();

        for (idx, item) in dx.indexed_iter_mut() {
            let y = self.out.as_ref().unwrap()[idx];

            *item = *item * (1.0 - y) * y;
        }

        dx
    }
}
