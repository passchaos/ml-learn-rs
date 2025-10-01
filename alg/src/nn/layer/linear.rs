use vectra::prelude::Matmul;

use crate::nn::{
    Ft, Mat,
    layer::LayerWard,
    matmul_policy,
    optimizer::{Optimizer, OptimizerOpT},
};

#[derive(Clone, Copy, Debug)]
pub enum WeightInit {
    Std(Ft),
    Xavier,
    He,
}

pub struct Linear {
    weight: Mat,
    bias: Option<Mat>,
    weight_opt: Optimizer,
    bias_opt: Option<Optimizer>,
    x: Option<Mat>,
}

impl Linear {
    pub fn new(
        weight_init: WeightInit,
        input_size: usize,
        output_size: usize,
        weight_opt: Optimizer,
        bias_opt: Option<Optimizer>,
    ) -> Self {
        let weight = Mat::randn([input_size, output_size]);

        let scale = match weight_init {
            WeightInit::Std(std) => std,
            WeightInit::Xavier => {
                let scale = (6.0 / (input_size + output_size) as Ft).sqrt();
                scale
            }
            WeightInit::He => {
                let scale = (2.0 / input_size as Ft).sqrt();
                scale
            }
        };

        let weight = weight.mul_scalar(scale);

        let bias = if bias_opt.is_some() {
            Some(Mat::zeros([1, output_size]))
        } else {
            None
        };

        Self {
            weight,
            bias,
            weight_opt,
            bias_opt,
            x: None,
        }
    }
}

impl LayerWard for Linear {
    fn forward(&mut self, input: &Mat) -> Mat {
        self.x = Some(input.clone());

        if let Some(bias) = &self.bias {
            &input.matmul(&self.weight, matmul_policy()) + bias
        } else {
            input.matmul(&self.weight, matmul_policy())
        }
    }

    fn backward(&mut self, grad: &Mat) -> Mat {
        let dx = grad.matmul(&self.weight.clone().transpose(), matmul_policy());

        let dw = self
            .x
            .as_ref()
            .unwrap()
            .clone()
            .transpose()
            .matmul(grad, matmul_policy());

        self.weight_opt.step(&mut self.weight, &dw);

        if let Some(bias) = self.bias.as_mut() {
            let db = grad.sum_axis(0);
            self.bias_opt.as_mut().unwrap().step(bias, &db);
        }

        dx
    }
}
