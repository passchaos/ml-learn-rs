use crate::nn::{
    Ft, Mat,
    layer::LayerWard,
    optimizer::{Optimizer, OptimizerOpT},
};

#[derive(Debug)]
struct ForwardInfo {
    batch_size: usize,
    xc: Mat,
    xn: Mat,
    std: Mat,
    running_mean: Mat,
    running_var: Mat,
}

#[derive(Debug)]
pub struct BatchNorm {
    gamma: Mat,
    beta: Mat,
    momentum: Ft,
    info: Option<ForwardInfo>,
    opt: Optimizer,
}

impl BatchNorm {
    pub fn new(gamma: Mat, beta: Mat, momentum: Ft, opt: Optimizer) -> Self {
        Self {
            gamma,
            beta,
            momentum,
            info: None,
            opt,
        }
    }
}

impl LayerWard for BatchNorm {
    fn forward(&mut self, input: &Mat) -> Mat {
        let mu = input.mean_axis(0);

        let xc = input - &mu;
        let var = xc.pow2().mean_axis(0);

        let std = var.clone().add_scalar(1.0e-6).sqrt();
        // let std = (&var + 1.0e-6).sqrt();

        let xn = &xc / &std;

        // println!("mu: {mu} xc: {xc} var: {var} std: {std} xn: {xn}");

        let (mut running_mean, mut running_var) = if let Some(info) = self.info.as_ref() {
            (info.running_mean.clone(), info.running_var.clone())
        } else {
            (Mat::zeros(input.shape()), Mat::zeros(input.shape()))
        };

        running_mean =
            &running_mean.mul_scalar(self.momentum) + &mu.mul_scalar(1.0 - self.momentum);
        running_var = &running_var.mul_scalar(self.momentum) + &var.mul_scalar(1.0 - self.momentum);

        let info = ForwardInfo {
            batch_size: input.shape()[0],
            xc: xc.clone(),
            xn: xn.clone(),
            std: std.clone(),
            running_mean,
            running_var,
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

    fn backward(&mut self, grad: &Mat) -> Mat {
        let Some(info) = self.info.as_ref() else {
            panic!("BatchNorm::backward called before forward");
        };

        let dbeta = grad.sum_axis(0);
        let dgemma = (&info.xn * grad).sum_axis(0);
        let dxn = &self.gamma * grad;
        let mut dxc = &dxn / &info.std;
        let dstd = (&(&dxn * &info.xc) / &(&info.std).pow2())
            .sum_axis(0)
            .mul_scalar(-1.0);
        let dvar = &dstd.mul_scalar(0.5) / &info.std;

        // println!("dbeta: {dbeta} dgamma: {dgemma} dxn: {dxn} dxc: {dxc} dstd: {dstd} dvar: {dvar}");

        dxc += &info.xc.clone().mul_scalar(2.0 / info.batch_size as Ft) * &dvar;

        let dmu = dxc.sum_axis(0);
        let dx = &dxc - &dmu.div_scalar(info.batch_size as Ft);

        self.opt.step(&mut self.gamma, &dgemma);
        self.opt.step(&mut self.beta, &dbeta);

        dx
    }
}

#[cfg(test)]
mod tests {
    use crate::nn::optimizer::Sgd;

    use super::*;

    #[test]
    fn test_batch_norm_forward() {
        let mut input = Mat::from_vec(
            vec![0.0, 0.2, 0.11, 0.13, 0.25, -0.02, 0.03, 0.23, 0.58, 0.19],
            [2, 5],
        );

        let gemma = Mat::ones([1, input.shape()[1]]);
        let beta = Mat::zeros([1, input.shape()[1]]);

        let mut batch_norm = BatchNorm::new(gemma, beta, 0.9, Optimizer::Sgd(Sgd::new(0.1)));

        for i in 0..5 {
            input = batch_norm.forward(&input);
            let res = batch_norm.backward(&input);
            println!("input{i}: {input} res{i}: {res}")
        }
    }
}
