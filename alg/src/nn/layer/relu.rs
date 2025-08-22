use burn_tensor::Bool;

use crate::nn::{Tensor2, layer::LayerWard};

#[derive(Default)]
pub struct Relu {
    mask: Option<Tensor2<Bool>>,
}

impl LayerWard for Relu {
    fn forward(&mut self, x: Tensor2) -> Tensor2 {
        if x.clone().contains_nan().into_scalar() == 1 {
            println!("relu meet nan value");
        }

        self.mask = Some(x.clone().lower_equal_elem(0.0));

        let out = x.clamp_min(0.0);

        out
    }

    fn backward(&mut self, grad: Tensor2) -> Tensor2 {
        let out = grad.mask_fill(self.mask.as_ref().unwrap().clone(), 0.0);

        out
    }
}
