use ndarray::Axis;

use crate::nn::{
    Float, Mat, Mat1, float_epsilon,
    layer::LayerWard,
    optimizer::{Optimizer, OptimizerOpT},
};

struct ForwardInfo {
    batch_size: usize,
    xc: Mat,
    xn: Mat,
    std: Mat1,
    running_mean: Mat,
    running_var: Mat,
}

pub struct BatchNorm {
    gamma: Mat,
    beta: Mat,
    momentum: Float,
    info: Option<ForwardInfo>,
    opt: Optimizer,
}

impl LayerWard for BatchNorm {
    fn forward(&mut self, input: &crate::nn::Mat) -> crate::nn::Mat {
        let mu = input.mean_axis(Axis(0)).unwrap();

        let xc = input - &mu;
        let var = xc.pow2().mean_axis(Axis(0)).unwrap();
        let std = (&var + float_epsilon()).sqrt();

        let xn = &xc / &std;

        let (mut running_mean, mut running_var) = if let Some(info) = self.info.as_ref() {
            (info.running_mean.clone(), info.running_var.clone())
        } else {
            (Mat::zeros(input.raw_dim()), Mat::zeros(input.raw_dim()))
        };

        running_mean = self.momentum * running_mean + (1.0 - self.momentum) * mu;
        running_var = self.momentum * running_var + (1.0 - self.momentum) * var;

        let info = ForwardInfo {
            batch_size: input.shape()[0],
            xc: xc.clone(),
            xn: xn.clone(),
            std: std.clone(),
            running_mean,
            running_var,
        };

        self.info = Some(info);

        &self.gamma * xn + &self.beta
    }

    fn backward(&mut self, grad: &crate::nn::Mat) -> crate::nn::Mat {
        let Some(info) = self.info.as_ref() else {
            panic!("BatchNorm::backward called before forward");
        };

        let dbeta = grad.sum_axis(Axis(0));
        let dgemma = (&info.xn * grad).sum_axis(Axis(0));
        let dxn = &self.gamma * grad;
        let mut dxc = &dxn / &info.std;
        let dstd = ((&dxn * &info.xc) / (&info.std).pow2()).sum_axis(Axis(0)) * -1.0;
        let dvar = 0.5 * &dstd / &info.std;
        dxc += &((2.0 / info.batch_size as Float) * &info.xc * &dvar);

        let dmu = dxc.sum_axis(Axis(0));
        let dx = (dxc - dmu) / info.batch_size as Float;

        let dgemma = dgemma.insert_axis(Axis(0));
        let dbeta = dbeta.insert_axis(Axis(0));

        self.opt.step(&mut self.gamma, &dgemma);
        self.opt.step(&mut self.beta, &dbeta);

        dx
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Axis, arr2};

    #[test]
    fn test_ndarray_axis() {
        let arr = arr2(&[[1, 2, 3], [4, 5, 6]]);
        let arr1 = arr.sum_axis(Axis(0));
        let arr2 = arr.sum_axis(Axis(0)).insert_axis(Axis(0));
        let arr3 = arr.sum_axis(Axis(0)).insert_axis(Axis(1));
        println!("arr= {arr} arr1= {arr1} arr2= {arr2} arr3= {arr3}");
    }
}
