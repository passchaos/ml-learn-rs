use ndarray::Axis;

use crate::{
    math::stat::randn,
    nn::{Float, Mat, layer::LayerWard, optimizer::Optimizer},
};

pub enum WeightInit {
    Std(Float),
    Xavier,
    He,
}

pub struct Linear {
    weight: Mat,
    bias: Option<Mat>,
    weight_opt: Box<dyn Optimizer>,
    bias_opt: Option<Box<dyn Optimizer>>,
    x: Option<Mat>,
}

impl Linear {
    pub fn new(
        weight_init: WeightInit,
        input_size: usize,
        output_size: usize,
        weight_opt: Box<dyn Optimizer>,
        bias_opt: Option<Box<dyn Optimizer>>,
        enable_bias: bool,
    ) -> Self {
        let weight = match weight_init {
            WeightInit::Std(std) => randn((output_size, input_size)) * std,
            WeightInit::Xavier => {
                let scale = (6.0 / (input_size + output_size) as Float).sqrt();
                randn((output_size, input_size)) * scale
            }
            WeightInit::He => {
                let scale = (2.0 / input_size as Float).sqrt();
                randn((output_size, input_size)) * scale
            }
        };

        let bias = if enable_bias {
            Some(Mat::zeros((output_size, 1)))
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
            self.weight.dot(input) + bias
        } else {
            self.weight.dot(input)
        }
    }

    fn backward(&mut self, grad: &Mat) -> Mat {
        let dx = grad.dot(&self.weight.t());

        let dw = self.x.as_ref().unwrap().t().dot(grad);

        self.weight_opt.step(&mut self.weight, &dw);

        if let Some(bias) = self.bias.as_mut() {
            let db = grad.sum_axis(Axis(0)).insert_axis(Axis(0));
            self.bias_opt.as_mut().unwrap().step(bias, &db);
        }

        dx
    }
}
