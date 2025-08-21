use crate::nn::{Tensor2, layer::LayerWard};

pub struct Sigmoid {
    out: Option<Tensor2>,
}

impl LayerWard for Sigmoid {
    fn forward(&mut self, input: Tensor2) -> Tensor2 {
        let out: Tensor2 = 1.0 / (1.0 + (-input).exp());

        self.out = Some(out.clone());

        out
    }

    fn backward(&mut self, grad: Tensor2) -> Tensor2 {
        let out = self.out.as_ref().unwrap().clone();
        let tmp = (1.0 - out.clone()) * out;

        grad * tmp
    }
}
