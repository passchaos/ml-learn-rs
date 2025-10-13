use std::fmt::Debug;

use num::Float;
use vectra::{NumExt, prelude::Array};

pub trait OptimizerOpT<const D: usize, T: Float + NumExt> {
    fn step(&mut self, param: &mut Array<D, T>, grad: &Array<D, T>);
}

#[derive(Clone, Debug)]
pub enum Optimizer<const D: usize, T: Debug> {
    Sgd(Sgd<T>),
    Momentum(Momentum<D, T>),
    AdaGrad(AdaGrad<D, T>),
    Adam(Adam<D, T>),
}

impl<const D: usize, T: Debug + Float + NumExt> OptimizerOpT<D, T> for Optimizer<D, T> {
    fn step(&mut self, param: &mut Array<D, T>, grad: &Array<D, T>) {
        match self {
            Optimizer::Sgd(sgd) => sgd.step(param, grad),
            Optimizer::Momentum(momentum) => momentum.step(param, grad),
            Optimizer::AdaGrad(adagrad) => adagrad.step(param, grad),
            Optimizer::Adam(adam) => adam.step(param, grad),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Sgd<T> {
    lr: T,
}

impl<T> Sgd<T> {
    pub fn new(lr: T) -> Self {
        Sgd { lr }
    }
}

impl<const D: usize, T: Float + NumExt> OptimizerOpT<D, T> for Sgd<T> {
    fn step(&mut self, param: &mut Array<D, T>, grad: &Array<D, T>) {
        *param -= grad.clone().mul_scalar(self.lr)
    }
}

#[derive(Clone, Debug)]
pub struct Momentum<const D: usize, T> {
    lr: T,
    momentum: T,
    v: Option<Array<D, T>>,
}

impl<const D: usize, T> Momentum<D, T> {
    pub fn new(lr: T, momentum: T) -> Self {
        Momentum {
            lr,
            momentum,
            v: None,
        }
    }
}

impl<const D: usize, T: Float + NumExt> OptimizerOpT<D, T> for Momentum<D, T> {
    fn step(&mut self, param: &mut Array<D, T>, grad: &Array<D, T>) {
        // 初始化v
        if self.v.is_none() {
            let default_value = Array::zeros(param.shape());

            self.v = Some(default_value);
        }

        // 动量处理
        let velocity = self.v.as_mut().unwrap();

        *velocity = &velocity.clone().mul_scalar(self.momentum) - &grad.clone().mul_scalar(self.lr);

        *param += &*velocity;
    }
}

#[derive(Clone, Debug)]
pub struct AdaGrad<const D: usize, T> {
    lr: T,
    h: Option<Array<D, T>>,
}

impl<const D: usize, T: Float + NumExt> AdaGrad<D, T> {
    pub fn new(lr: T) -> Self {
        Self { lr, h: None }
    }
}

impl<const D: usize, T: Float + NumExt> OptimizerOpT<D, T> for AdaGrad<D, T> {
    fn step(&mut self, param: &mut Array<D, T>, grad: &Array<D, T>) {
        // 初始化v
        if self.h.is_none() {
            let default_value = Array::zeros(param.shape());

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
pub struct RMSprop<const D: usize, T: Float + NumExt> {
    lr: T,
    decay_rate: T,
    h: Option<Array<D, T>>,
}

impl<const D: usize, T: Float + NumExt> RMSprop<D, T> {
    pub fn new(lr: T, decay_rate: T) -> Self {
        Self {
            lr,
            decay_rate,
            h: None,
        }
    }
}

impl<const D: usize, T: Float + NumExt> OptimizerOpT<D, T> for RMSprop<D, T> {
    fn step(&mut self, param: &mut Array<D, T>, grad: &Array<D, T>) {
        // 初始化v
        if self.h.is_none() {
            let default_value = Array::zeros(param.shape());

            self.h = Some(default_value);
        }

        let h_value = self.h.as_mut().unwrap();

        *h_value = h_value.clone().mul_scalar(self.decay_rate);

        *h_value = grad.pow2().mul_scalar(T::one() - self.decay_rate);

        let g_value = grad.clone().mul_scalar(self.lr);
        let h_value = h_value.sqrt().add_scalar(crate::nn::delta());

        let new = &g_value / &h_value;
        *param -= new;
    }
}

#[derive(Clone, Debug)]
pub struct Adam<const D: usize, T> {
    lr: T,
    beta1: T,
    beta2: T,
    iter: i32,
    m: Option<Array<D, T>>,
    v: Option<Array<D, T>>,
}

impl<const D: usize, T: Float + NumExt> Adam<D, T> {
    pub fn new(lr: T, beta1: T, beta2: T) -> Self {
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

impl<const D: usize, T: Float + NumExt> OptimizerOpT<D, T> for Adam<D, T> {
    fn step(&mut self, param: &mut Array<D, T>, grad: &Array<D, T>) {
        if self.m.is_none() {
            let default_value = Array::zeros(param.shape());

            self.m = Some(default_value.clone());
            self.v = Some(default_value);
        }

        self.iter += 1;

        let lr_t = self.lr * (T::one() - self.beta2.powi(self.iter)).sqrt()
            / (T::one() - self.beta1.powi(self.iter));

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        let m_value = (grad - &*m).mul_scalar(T::one() - self.beta1);
        *m += &m_value;
        let v_value = (&grad.pow2() - &*v).mul_scalar(T::one() - self.beta2);
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
