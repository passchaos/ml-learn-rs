use std::fmt::Debug;

use num::Float;
use num::cast;
use vectra::{NumExt, prelude::Array};

use crate::nn::{
    delta,
    layer::LayerWard,
    optimizer::{Optimizer, OptimizerOpT},
};

#[derive(Debug)]
struct ForwardInfo<const D: usize, T: Debug> {
    batch_size: usize,
    xc: Array<2, T>,
    xn: Array<2, T>,
    std: Array<2, T>,
    running_mean: Array<2, T>,
    running_var: Array<2, T>,
    input_shape: [usize; D],
}

#[derive(Debug)]
pub struct BatchNorm<const D: usize, T: Debug> {
    gamma: Array<2, T>,
    beta: Array<2, T>,
    momentum: T,
    info: Option<ForwardInfo<D, T>>,
    opt: Optimizer<2, T>,
}

impl<const D: usize, T: Debug> BatchNorm<D, T> {
    pub fn new(gamma: Array<2, T>, beta: Array<2, T>, momentum: T, opt: Optimizer<2, T>) -> Self {
        Self {
            gamma,
            beta,
            momentum,
            info: None,
            opt,
        }
    }
}

impl<const D: usize, T: Debug + Float + NumExt> LayerWard<D, 2, T> for BatchNorm<D, T> {
    fn forward(&mut self, input: &Array<D, T>) -> Array<2, T> {
        let input_shape = input.shape();
        let batch_size = input_shape[0];

        let input = input.clone().reshape([batch_size as isize, -1]);

        let mu = input.mean_axis(0);

        let xc = &input - &mu;
        let var = xc.pow2().mean_axis(0);

        let std = var.clone().add_scalar(delta()).sqrt();
        // let std = (&var + 1.0e-6).sqrt();

        let xn = &xc / &std;

        // println!("mu: {mu} xc: {xc} var: {var} std: {std} xn: {xn}");

        let (mut running_mean, mut running_var) = if let Some(info) = self.info.as_ref() {
            (info.running_mean.clone(), info.running_var.clone())
        } else {
            (
                Array::<2, T>::zeros(input.shape()),
                Array::<2, T>::zeros(input.shape()),
            )
        };

        running_mean =
            &running_mean.mul_scalar(self.momentum) + &mu.mul_scalar(T::one() - self.momentum);
        running_var =
            &running_var.mul_scalar(self.momentum) + &var.mul_scalar(T::one() - self.momentum);

        let info = ForwardInfo {
            batch_size,
            xc: xc.clone(),
            xn: xn.clone(),
            std: std.clone(),
            running_mean,
            running_var,
            input_shape,
        };

        self.info = Some(info);
        // println!(
        //     "shape: input= {:?} gamma= {:?} xn= {:?} beta= {:?}",
        //     input.shape(),
        //     self.gamma.shape(),
        //     xn.shape(),
        //     self.beta.shape()
        // );

        &(&self.gamma * &xn) + &self.beta
    }

    fn backward(&mut self, grad: &Array<2, T>) -> Array<D, T> {
        let Some(info) = self.info.as_ref() else {
            panic!("BatchNorm::backward called before forward");
        };

        let dbeta = grad.sum_axis(0);
        let dgemma = (&info.xn * grad).sum_axis(0);
        let dxn = &self.gamma * grad;
        let mut dxc = &dxn / &info.std;
        let dstd = (&(&dxn * &info.xc) / &(&info.std).pow2())
            .sum_axis(0)
            .mul_scalar(-T::one());
        let dvar = &dstd.mul_scalar(cast(0.5).unwrap()) / &info.std;

        // println!("dbeta: {dbeta} dgamma: {dgemma} dxn: {dxn} dxc: {dxc} dstd: {dstd} dvar: {dvar}");

        dxc += &info
            .xc
            .clone()
            .mul_scalar(cast::<_, T>(2.0).unwrap() / cast(info.batch_size).unwrap())
            * &dvar;

        let dmu = dxc.sum_axis(0);
        let dx = &dxc - &dmu.div_scalar(cast(info.batch_size).unwrap());

        self.opt.step(&mut self.gamma, &dgemma);
        self.opt.step(&mut self.beta, &dbeta);

        dx.reshape(self.info.as_ref().unwrap().input_shape.map(|i| i as isize))
    }
}

#[cfg(test)]
mod tests {
    use crate::nn::optimizer::Sgd;

    use super::*;

    #[test]
    fn test_batch_norm_forward() {
        let mut input = Array::<2, f64>::from_vec(
            vec![0.0, 0.2, 0.11, 0.13, 0.25, -0.02, 0.03, 0.23, 0.58, 0.19],
            [2, 5],
        );

        let gemma = Array::<2, f64>::ones([1, input.shape()[1]]);
        let beta = Array::<2, f64>::zeros([1, input.shape()[1]]);

        let mut batch_norm = BatchNorm::new(gemma, beta, 0.9, Optimizer::Sgd(Sgd::new(0.1)));

        for i in 0..5 {
            input = batch_norm.forward(&input);
            let res = batch_norm.backward(&input);
            println!("input{i}: {input} res{i}: {res}")
        }
    }
}
