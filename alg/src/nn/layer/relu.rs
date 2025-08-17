use crate::nn::{Mat, layer::LayerWard};

#[derive(Default)]
pub struct Relu {
    mask: Option<Mat<bool>>,
}

impl LayerWard for Relu {
    fn forward(&mut self, x: &Mat) -> Mat {
        self.mask = Some(x.mapv(|a| a <= 0.0));

        let out = x.mapv(|a| if a > 0.0 { a } else { 0.0 });

        out
    }

    fn backward(&mut self, grad: &Mat) -> Mat {
        let mut out = grad.clone();

        for (idx, a) in out.indexed_iter_mut() {
            let v = self.mask.as_ref().unwrap()[idx];

            if v {
                *a = 0.0;
            }
        }

        out
    }
}
