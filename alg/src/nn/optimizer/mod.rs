use crate::nn::Ft;
use crate::nn::Mat;

pub trait OptimizerOpT: std::fmt::Debug {
    fn step(&mut self, param: &mut Mat, grad: &Mat);
}

#[derive(Clone, Debug)]
pub enum Optimizer {
    Sgd(Sgd),
    Momentum(Momentum),
    AdaGrad(AdaGrad),
    Adam(Adam),
}

impl OptimizerOpT for Optimizer {
    fn step(&mut self, param: &mut Mat, grad: &Mat) {
        match self {
            Optimizer::Sgd(sgd) => sgd.step(param, grad),
            Optimizer::Momentum(momentum) => momentum.step(param, grad),
            Optimizer::AdaGrad(adagrad) => adagrad.step(param, grad),
            Optimizer::Adam(adam) => adam.step(param, grad),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Sgd {
    lr: Ft,
}

impl Sgd {
    pub fn new(lr: Ft) -> Self {
        Sgd { lr }
    }
}

impl OptimizerOpT for Sgd {
    fn step(&mut self, param: &mut Mat, grad: &Mat) {
        *param -= grad.clone().mul_scalar(self.lr)
    }
}

#[derive(Clone, Debug)]
pub struct Momentum {
    lr: Ft,
    momentum: Ft,
    v: Option<Mat>,
}

impl Momentum {
    pub fn new(lr: Ft, momentum: Ft) -> Self {
        Momentum {
            lr,
            momentum,
            v: None,
        }
    }
}

impl OptimizerOpT for Momentum {
    fn step(&mut self, param: &mut Mat, grad: &Mat) {
        // 初始化v
        if self.v.is_none() {
            let default_value = Mat::zeros(param.shape());

            self.v = Some(default_value);
        }

        // 动量处理
        let velocity = self.v.as_mut().unwrap();

        *velocity = &velocity.clone().mul_scalar(self.momentum) - &grad.clone().mul_scalar(self.lr);

        *param += &*velocity;
    }
}

#[derive(Clone, Debug)]
pub struct AdaGrad {
    lr: Ft,
    h: Option<Mat>,
}

impl AdaGrad {
    pub fn new(lr: Ft) -> Self {
        Self { lr, h: None }
    }
}

impl OptimizerOpT for AdaGrad {
    fn step(&mut self, param: &mut Mat, grad: &Mat) {
        // 初始化v
        if self.h.is_none() {
            let default_value = Mat::zeros(param.shape());

            self.h = Some(default_value);
        }

        let h_value = self.h.as_mut().unwrap();
        *h_value += &(grad * grad);

        let g_value = grad.clone().mul_scalar(self.lr);
        let h_value = h_value.sqrt().add_scalar(crate::nn::delta());

        let new = &g_value / &h_value;
        *param -= new;
    }
}

#[derive(Clone, Debug)]
pub struct RMSprop {
    lr: Ft,
    decay_rate: Ft,
    h: Option<Mat>,
}

impl RMSprop {
    pub fn new(lr: Ft, decay_rate: Ft) -> Self {
        Self {
            lr,
            decay_rate,
            h: None,
        }
    }
}

impl OptimizerOpT for RMSprop {
    fn step(&mut self, param: &mut Mat, grad: &Mat) {
        // 初始化v
        if self.h.is_none() {
            let default_value = Mat::zeros(param.shape());

            self.h = Some(default_value);
        }

        let h_value = self.h.as_mut().unwrap();

        *h_value = h_value.clone().mul_scalar(self.decay_rate);

        *h_value = grad.pow2().mul_scalar(1.0 - self.decay_rate);

        let g_value = grad.clone().mul_scalar(self.lr);
        let h_value = h_value.sqrt().add_scalar(crate::nn::delta());

        let new = &g_value / &h_value;
        *param -= new;
    }
}

#[derive(Clone, Debug)]
pub struct Adam {
    lr: Ft,
    beta1: Ft,
    beta2: Ft,
    iter: i32,
    m: Option<Mat>,
    v: Option<Mat>,
}

impl Adam {
    pub fn new(lr: Ft, beta1: Ft, beta2: Ft) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            iter: 0,
            m: None,
            v: None,
        }
    }
}

impl OptimizerOpT for Adam {
    fn step(&mut self, param: &mut Mat, grad: &Mat) {
        if self.m.is_none() {
            let default_value = Mat::zeros(param.shape());

            self.m = Some(default_value.clone());
            self.v = Some(default_value);
        }

        self.iter += 1;

        let lr_t = self.lr * (1.0 - self.beta2.powi(self.iter)).sqrt()
            / (1.0 - self.beta1.powi(self.iter));

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        let m_value = (grad - &*m).mul_scalar(1.0 - self.beta1);
        *m += &m_value;
        let v_value = (&grad.pow2() - &*v).mul_scalar(1.0 - self.beta2);
        *v += &v_value;

        let v1 = m.clone().mul_scalar(lr_t);
        let v2 = v.sqrt().add_scalar(crate::nn::delta());
        *param -= &v1 / &v2;
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use vectra::prelude::Array;

    use super::*;

    #[test]
    fn test_sgd() {
        let mut optimizer = Sgd::new(0.01);

        let mut param = Array::from_vec(
            vec![2.3012, 1.3212, 2.21212, 12.2, -2.1212, 32.122, 20.22, 0.11],
            [2, 4],
        );

        let grad = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0], [2, 4]);

        optimizer.step(&mut param, &grad);

        let result = Array::from_vec(
            vec![
                2.2912, 1.3012, 2.18212, 12.16, -2.1112, 32.142002, 20.25, 0.15,
            ],
            [2, 4],
        );

        assert_relative_eq!(param, result);
    }
}
