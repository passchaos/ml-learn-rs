use crate::nn::Float;
use crate::nn::Mat;

pub trait Optimizer {
    fn step(&mut self, param: &mut Mat, grad: &Mat);
}

#[derive(Debug)]
pub struct Sgd {
    lr: Float,
}

impl Sgd {
    pub fn new(lr: Float) -> Self {
        Sgd { lr }
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, param: &mut Mat, grad: &Mat) {
        *param -= &(grad * self.lr.clone());
    }
}

#[derive(Debug)]
pub struct Momentum {
    lr: Float,
    momentum: Float,
    v: Option<Mat>,
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

impl Optimizer for Momentum {
    fn step(&mut self, param: &mut Mat, grad: &Mat) {
        // 初始化v
        if self.v.is_none() {
            let default_value = Mat::zeros(param.raw_dim());

            self.v = Some(default_value);
        }

        // 动量处理
        let velocity = self.v.as_mut().unwrap();
        *velocity = &(&*velocity * self.momentum.clone()) - &(grad * self.lr.clone());

        *param += &*velocity;
    }
}

#[derive(Debug)]
pub struct AdaGrad {
    lr: Float,
    h: Option<Mat>,
}

impl AdaGrad {
    pub fn new(lr: Float) -> Self {
        Self { lr, h: None }
    }
}

impl Optimizer for AdaGrad {
    fn step(&mut self, param: &mut Mat, grad: &Mat) {
        // 初始化v
        if self.h.is_none() {
            let default_value = Mat::zeros(param.raw_dim());

            self.h = Some(default_value);
        }

        let h_value = self.h.as_mut().unwrap();
        *h_value += &(grad * grad);

        let g_value = grad * self.lr.clone();
        let h_value = h_value.sqrt() + crate::nn::float_epsilon();

        let new = &g_value / &h_value;
        *param -= &new;
    }
}

#[derive(Debug)]
pub struct RMSprop {
    lr: Float,
    decay_rate: Float,
    h: Option<Mat>,
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

impl Optimizer for RMSprop {
    fn step(&mut self, param: &mut Mat, grad: &Mat) {
        // 初始化v
        if self.h.is_none() {
            let default_value = Mat::zeros(param.raw_dim());

            self.h = Some(default_value);
        }

        let h_value = self.h.as_mut().unwrap();

        *h_value *= self.decay_rate;
        *h_value += &(grad * grad * (1.0 - self.decay_rate));

        let g_value = grad * self.lr.clone();
        let h_value = h_value.sqrt() + crate::nn::float_epsilon();

        let new = &g_value / &h_value;
        *param -= &new;
    }
}

pub struct Adam {
    lr: Float,
    beta1: Float,
    beta2: Float,
    iter: i32,
    m: Option<Mat>,
    v: Option<Mat>,
}

impl Optimizer for Adam {
    fn step(&mut self, param: &mut Mat, grad: &Mat) {
        if self.m.is_none() {
            let default_value = Mat::zeros(param.raw_dim());

            self.m = Some(default_value.clone());
            self.v = Some(default_value);
        }

        self.iter += 1;

        let lr_t =
            self.lr * (1.0 - self.beta2.powi(self.iter)) / (1.0 - self.beta1.powi(self.iter));

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        let m_value = (grad - &*m) * (1.0 - self.beta1);
        *m += &m_value;
        let v_value = (grad.pow2() - &*v) * (1.0 - self.beta2);
        *v += &v_value;

        let v1 = &*m * lr_t;
        let v2 = v.sqrt() - crate::nn::float_epsilon();
        *param -= &(&v1 / &v2);
    }
}
