use crate::nn::Float;
use crate::nn::Tensor2;
use crate::nn::default_device;

pub trait OptimizerOpT: std::fmt::Debug {
    fn step(&mut self, param: &mut Tensor2, grad: Tensor2);
}

#[derive(Clone, Debug)]
pub enum Optimizer {
    Sgd(Sgd),
    Momentum(Momentum),
    AdaGrad(AdaGrad),
    Adam(Adam),
}

impl OptimizerOpT for Optimizer {
    fn step(&mut self, param: &mut Tensor2, grad: Tensor2) {
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
    lr: Float,
}

impl Sgd {
    pub fn new(lr: Float) -> Self {
        Sgd { lr }
    }
}

impl OptimizerOpT for Sgd {
    fn step(&mut self, param: &mut Tensor2, grad: Tensor2) {
        *param = param.clone() - grad * self.lr;
    }
}

#[derive(Clone, Debug)]
pub struct Momentum {
    lr: Float,
    momentum: Float,
    v: Option<Tensor2>,
}

impl Momentum {
    pub fn new(lr: Float, momentum: Float) -> Self {
        Momentum {
            lr,
            momentum,
            v: None,
        }
    }
}

impl OptimizerOpT for Momentum {
    fn step(&mut self, param: &mut Tensor2, grad: Tensor2) {
        // 初始化v
        if self.v.is_none() {
            let default_value = Tensor2::zeros(param.shape(), &default_device());

            self.v = Some(default_value);
        }

        // 动量处理
        let velocity = self.v.as_mut().unwrap();
        *velocity = velocity.clone() * self.momentum - grad * self.lr;

        *param = param.clone() + velocity.clone();
    }
}

#[derive(Clone, Debug)]
pub struct AdaGrad {
    lr: Float,
    h: Option<Tensor2>,
}

impl AdaGrad {
    pub fn new(lr: Float) -> Self {
        Self { lr, h: None }
    }
}

impl OptimizerOpT for AdaGrad {
    fn step(&mut self, param: &mut Tensor2, grad: Tensor2) {
        // 初始化v
        if self.h.is_none() {
            let default_value = Tensor2::zeros(param.shape(), &default_device());

            self.h = Some(default_value);
        }

        let h_value = self.h.as_mut().unwrap();
        *h_value = h_value.clone() + grad.clone().powi_scalar(2);

        let g_value = grad * self.lr;
        let h_value = h_value.clone().sqrt() + crate::nn::float_epsilon();

        let new = g_value / h_value;
        *param = param.clone() - new;
    }
}

#[derive(Clone, Debug)]
pub struct RMSprop {
    lr: Float,
    decay_rate: Float,
    h: Option<Tensor2>,
}

impl RMSprop {
    pub fn new(lr: Float, decay_rate: Float) -> Self {
        Self {
            lr,
            decay_rate,
            h: None,
        }
    }
}

impl OptimizerOpT for RMSprop {
    fn step(&mut self, param: &mut Tensor2, grad: Tensor2) {
        // 初始化v
        if self.h.is_none() {
            let default_value = Tensor2::zeros(param.shape(), &default_device());

            self.h = Some(default_value);
        }

        let h_value = self.h.as_mut().unwrap();

        *h_value = h_value.clone() * self.decay_rate;
        *h_value = h_value.clone() + grad.clone().powi_scalar(2) * (1.0 - self.decay_rate);

        let g_value = grad * self.lr.clone();
        let h_value = h_value.clone().sqrt() + crate::nn::float_epsilon();

        let new = g_value / h_value;
        *param = param.clone() - new;
    }
}

#[derive(Clone, Debug)]
pub struct Adam {
    lr: Float,
    beta1: Float,
    beta2: Float,
    iter: i32,
    m: Option<Tensor2>,
    v: Option<Tensor2>,
}

impl Adam {
    pub fn new(lr: Float, beta1: Float, beta2: Float) -> Self {
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
    fn step(&mut self, param: &mut Tensor2, grad: Tensor2) {
        if self.m.is_none() {
            let default_value = Tensor2::zeros(param.shape(), &default_device());

            self.m = Some(default_value.clone());
            self.v = Some(default_value);
        }

        self.iter += 1;

        let lr_t =
            self.lr * (1.0 - self.beta2.powi(self.iter)) / (1.0 - self.beta1.powi(self.iter));

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        let m_value = (grad.clone() - m.clone()) * (1.0 - self.beta1);
        *m = m.clone() + m_value;
        let v_value = (grad.powi_scalar(2) - v.clone()) * (1.0 - self.beta2);
        *v = v.clone() + v_value;

        let v1 = m.clone() * lr_t;
        let v2 = v.clone().sqrt() + crate::nn::float_epsilon();
        *param = param.clone() - v1 / v2;
    }
}
