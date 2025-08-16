use ndarray::Axis;

use crate::{
    math::stat::randn,
    nn::{Mat, Mat1, layer::Layer, optimizer::Optimizer},
};

pub struct Linear {
    weight: Mat,
    bias: Option<Mat>,
    x: Option<Mat>,
}

impl Linear {
    pub fn new(input_dim: usize, output_dim: usize, enable_bias: bool) -> Self {
        let weight = randn((output_dim, input_dim));

        let bias = if enable_bias {
            Some(Mat::zeros((output_dim, 1)))
        } else {
            None
        };

        Self {
            weight,
            bias,
            x: None,
        }
    }
}

impl Layer for Linear {
    fn forward(&mut self, input: &Mat) -> Mat {
        self.x = Some(input.clone());

        if let Some(bias) = &self.bias {
            self.weight.dot(input) + bias
        } else {
            self.weight.dot(input)
        }
    }

    fn backward<O: Optimizer>(&mut self, grad: &Mat, opt: &mut O) -> Mat {
        let dx = grad.dot(&self.weight.t());

        let dw = self.x.as_ref().unwrap().t().dot(grad);
        opt.step(&mut self.weight, &dw);

        if let Some(bias) = self.bias.as_mut() {
            let db = grad.sum_axis(Axis(0)).insert_axis(Axis(0));
            opt.step(bias, &db);
        }

        dx
    }
}
