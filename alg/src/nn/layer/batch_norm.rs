use ndarray::Axis;

use crate::nn::{float_epsilon, layer::LayerWard};

pub struct BatchNorm {}

impl LayerWard for BatchNorm {
    fn forward(&mut self, input: &crate::nn::Mat) -> crate::nn::Mat {
        let mu = input.mean_axis(Axis(0)).unwrap();

        let xc = input - &mu;
        let var = xc.pow2().mean_axis(Axis(0)).unwrap();
        let std = (var + float_epsilon()).sqrt();

        let xn = &xc / &std;

        xn
    }

    fn backward(&mut self, grad: &crate::nn::Mat) -> crate::nn::Mat {
        unimplemented!()
    }
}
