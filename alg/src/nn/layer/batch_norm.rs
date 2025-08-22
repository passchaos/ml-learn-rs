use crate::nn::{
    Float, Tensor2, default_device,
    layer::LayerWard,
    optimizer::{Optimizer, OptimizerOpT},
};

struct ForwardInfo {
    batch_size: usize,
    xc: Tensor2,
    xn: Tensor2,
    std: Tensor2,
    running_mean: Tensor2,
    running_var: Tensor2,
}

pub struct BatchNorm {
    gamma: Tensor2,
    beta: Tensor2,
    momentum: Float,
    info: Option<ForwardInfo>,
    opt: Optimizer,
}

impl BatchNorm {
    pub fn new(gamma: Tensor2, beta: Tensor2, momentum: Float, opt: Optimizer) -> Self {
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
    fn forward(&mut self, input: crate::nn::Tensor2) -> crate::nn::Tensor2 {
        // if input.clone().contains_nan().into_scalar() == 1 {
        //     println!("batch norm meet nan value");
        // }

        let mu = input.clone().mean_dim(0);
        // let mu = input.mean_axis(Axis(0)).unwrap();

        let xc = input.clone() - mu.clone();

        let var = xc.clone().powi_scalar(2).mean_dim(0);
        let std = (var.clone() + crate::nn::float_epsilon()).sqrt();

        let xn = xc.clone() / std.clone();

        // println!("mu: {mu} xc: {xc} var: {var} std: {std} xn: {xn}");

        let (mut running_mean, mut running_var) = if let Some(info) = self.info.as_ref() {
            (info.running_mean.clone(), info.running_var.clone())
        } else {
            (
                Tensor2::zeros(input.shape(), &default_device()),
                Tensor2::zeros(input.shape(), &default_device()),
            )
        };

        running_mean = self.momentum * running_mean + (1.0 - self.momentum) * mu;
        running_var = self.momentum * running_var + (1.0 - self.momentum) * var;

        let info = ForwardInfo {
            batch_size: input.dims()[0],
            xc,
            xn: xn.clone(),
            std,
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

        self.gamma.clone() * xn + self.beta.clone()
    }

    fn backward(&mut self, grad: crate::nn::Tensor2) -> crate::nn::Tensor2 {
        let Some(info) = self.info.as_ref() else {
            panic!("BatchNorm::backward called before forward");
        };

        let dbeta = grad.clone().sum_dim(0);
        let dgemma = (info.xn.clone() * grad.clone()).sum_dim(0);
        let dxn = self.gamma.clone() * grad;
        let mut dxc = dxn.clone() / info.std.clone();
        let dstd = ((dxn * info.xc.clone()) / info.std.clone().powf_scalar(2)).sum_dim(0) * -1.0;
        let dvar = 0.5 * dstd / info.std.clone();

        // println!("dbeta: {dbeta} dgamma: {dgemma} dxn: {dxn} dxc: {dxc} dstd: {dstd} dvar: {dvar}");

        dxc = dxc + (2.0 / info.batch_size as Float) * info.xc.clone() * dvar;

        let dmu = dxc.clone().sum_dim(0);
        let dx = dxc - (dmu / info.batch_size as Float);

        self.opt.step(&mut self.gamma, dgemma);
        self.opt.step(&mut self.beta, dbeta);

        dx
    }
}

#[cfg(test)]
mod tests {
    use crate::nn::optimizer::Sgd;

    use super::*;
    use ndarray::{Axis, arr2};

    #[test]
    fn test_ndarray_axis() {
        let arr = arr2(&[[1, 2, 3], [4, 5, 6]]);
        let arr1 = arr.sum_axis(Axis(0));
        let arr2 = arr.sum_axis(Axis(0)).insert_axis(Axis(0));
        let arr3 = arr.sum_axis(Axis(0)).insert_axis(Axis(1));
        println!("arr= {arr} arr1= {arr1} arr2= {arr2} arr3= {arr3}");
    }

    #[test]
    fn test_batch_norm_forward() {
        let mut input = Tensor2::from_data(
            [
                [0.0, 0.2, 0.11, 0.13, 0.25],
                [-0.02, 0.03, 0.23, 0.58, 0.19],
            ],
            &default_device(),
        );

        let gemma = Tensor2::ones([1, input.dims()[1]], &default_device());
        let beta = Tensor2::zeros([1, input.dims()[1]], &default_device());

        let mut batch_norm = BatchNorm::new(gemma, beta, 0.9, Optimizer::Sgd(Sgd::new(0.1)));

        for i in 0..5 {
            input = batch_norm.forward(input.clone());
            let res = batch_norm.backward(input.clone());
            println!("input{i}: {input} res{i}: {res}")
        }
    }
}
