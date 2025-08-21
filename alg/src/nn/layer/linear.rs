use ndarray::Axis;

use crate::{
    math::stat::randn,
    nn::{
        Float, Tensor2, default_device,
        layer::LayerWard,
        optimizer::{Optimizer, OptimizerOpT},
    },
};

#[derive(Clone, Copy, Debug)]
pub enum WeightInit {
    Std(Float),
    Xavier,
    He,
}

pub struct Linear {
    weight: Tensor2,
    bias: Option<Tensor2>,
    weight_opt: Optimizer,
    bias_opt: Option<Optimizer>,
    x: Option<Tensor2>,
}

impl Linear {
    pub fn new(
        weight_init: WeightInit,
        input_size: usize,
        output_size: usize,
        weight_opt: Optimizer,
        bias_opt: Option<Optimizer>,
    ) -> Self {
        let weight = Tensor2::random(
            [input_size, output_size],
            burn_tensor::Distribution::Normal(0.0, 1.0),
            &default_device(),
        );

        let scale = match weight_init {
            WeightInit::Std(std) => std,
            WeightInit::Xavier => {
                let scale = (6.0 / (input_size + output_size) as Float).sqrt();
                scale
            }
            WeightInit::He => {
                let scale = (2.0 / input_size as Float).sqrt();
                scale
            }
        };

        let weight = weight * scale;

        let bias = if bias_opt.is_some() {
            Some(Tensor2::zeros([1, output_size], &default_device()))
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
    fn forward(&mut self, input: Tensor2) -> Tensor2 {
        self.x = Some(input.clone());

        if let Some(bias) = &self.bias {
            input.matmul(self.weight.clone()) + bias.clone()
        } else {
            input.matmul(self.weight.clone())
        }
    }

    fn backward(&mut self, grad: Tensor2) -> Tensor2 {
        let dx = grad.clone().matmul(self.weight.clone().transpose());

        let dw = self
            .x
            .as_ref()
            .unwrap()
            .clone()
            .transpose()
            .matmul(grad.clone());

        self.weight_opt.step(&mut self.weight, dw);

        if let Some(bias) = self.bias.as_mut() {
            let db = grad.sum_dim(0);
            self.bias_opt.as_mut().unwrap().step(bias, db);
        }

        dx
    }
}
