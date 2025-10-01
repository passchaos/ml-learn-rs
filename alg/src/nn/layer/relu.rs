use crate::nn::{Mat, layer::LayerWard};

#[derive(Default)]
pub struct Relu {
    mask: Option<Mat<bool>>,
}

impl LayerWard for Relu {
    fn forward(&mut self, x: &Mat) -> Mat {
        self.mask = Some(x.map(|&a| a <= 0.0));

        let out = x.map(|&a| if a > 0.0 { a } else { 0.0 });

        out
    }

    fn backward(&mut self, grad: &Mat) -> Mat {
        let mut out = grad.clone();

        out.multi_iter_mut(|idx, item| {
            let v = self.mask.as_ref().unwrap()[idx.map(|a| a as isize)];

            if v {
                *item = 0.0;
            }
        });

        out
    }
}
