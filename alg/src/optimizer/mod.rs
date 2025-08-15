use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Sub;
use std::ops::SubAssign;

use ndarray::Array;
use ndarray::ArrayBase;
use ndarray::Data;
use ndarray::DataMut;
use ndarray::DimMax;
use ndarray::Dimension;
use ndarray::RawData;
use ndarray::ScalarOperand;
use num::Float;
use num::Zero;

pub trait Optimizer<SL: RawData, SR: RawData, DL, DR> {
    fn step(&mut self, param: &mut ArrayBase<SL, DL>, grad: &ArrayBase<SR, DR>);
}

#[derive(Debug)]
pub struct Sgd<T> {
    lr: T,
}

impl<T> Sgd<T> {
    pub fn new(lr: T) -> Self {
        Sgd { lr }
    }
}

impl<
    SL: DataMut<Elem = E>,
    SR: Data<Elem = E>,
    E: Clone + Mul<T, Output = E> + SubAssign<E>,
    DL: Dimension,
    DR: Dimension,
    T: ScalarOperand,
> Optimizer<SL, SR, DL, DR> for Sgd<T>
{
    fn step(&mut self, param: &mut ArrayBase<SL, DL>, grad: &ArrayBase<SR, DR>) {
        *param -= &(grad * self.lr.clone());
    }
}

#[derive(Debug)]
pub struct Momentum<T, V: std::fmt::Debug, VD: Dimension> {
    lr: T,
    momentum: T,
    v: Option<Array<V, VD>>,
}

impl<T, V: std::fmt::Debug, VD: Dimension> Momentum<T, V, VD> {
    pub fn new(lr: T, momentum: T) -> Self {
        Momentum {
            lr,
            momentum,
            v: None,
        }
    }
}

impl<SL, SR, DL, DR, E, T> Optimizer<SL, SR, DL, DR> for Momentum<T, E, DL>
where
    SL: DataMut<Elem = E>,
    SR: Data<Elem = E>,
    E: Clone + std::fmt::Debug + Mul<T, Output = E> + Sub<E, Output = E> + AddAssign<E> + Zero,
    DL: Dimension + DimMax<DR, Output = DL>,
    DR: Dimension,
    T: ScalarOperand,
{
    fn step(&mut self, param: &mut ArrayBase<SL, DL>, grad: &ArrayBase<SR, DR>) {
        // 初始化v
        if self.v.is_none() {
            let default_value = Array::zeros::<DL>(param.raw_dim());

            self.v = Some(default_value);
        }

        // 动量处理
        let velocity = self.v.as_mut().unwrap();
        *velocity = &(&*velocity * self.momentum.clone()) - &(grad * self.lr.clone());

        *param += velocity;
    }
}

#[derive(Debug)]
pub struct AdaGrad<T, V: std::fmt::Debug, VD: Dimension> {
    lr: T,
    h: Option<Array<V, VD>>,
}

impl<T, V: std::fmt::Debug, VD: Dimension> AdaGrad<T, V, VD> {
    pub fn new(lr: T) -> Self {
        Self { lr, h: None }
    }
}

impl<SL, SR, DL, DR, E, T> Optimizer<SL, SR, DL, DR> for AdaGrad<T, E, DL>
where
    SL: DataMut<Elem = E>,
    SR: Data<Elem = E>,
    E: Clone
        + std::fmt::Debug
        + Zero
        + AddAssign<E>
        + Mul<E, Output = E>
        + Mul<T, Output = E>
        + 'static
        + Float
        + Add<E, Output = E>
        + ScalarOperand
        + SubAssign<E>,
    DL: Dimension,
    DR: Dimension + DimMax<DL>,
    T: ScalarOperand,
{
    fn step(&mut self, param: &mut ArrayBase<SL, DL>, grad: &ArrayBase<SR, DR>) {
        // 初始化v
        if self.h.is_none() {
            let default_value = Array::zeros::<DL>(param.raw_dim());

            self.h = Some(default_value);
        }

        let h_value = self.h.as_mut().unwrap();
        *h_value += &(grad * grad);

        let g_value = grad * self.lr.clone();
        let h_value = h_value.sqrt() + E::epsilon();

        let new = &g_value / &h_value;
        *param -= &new;
    }
}

#[derive(Debug)]
pub struct RMSprop<T, V: std::fmt::Debug, VD: Dimension> {
    lr: T,
    decay_rate: T,
    h: Option<Array<V, VD>>,
}

impl<T, V: std::fmt::Debug, VD: Dimension> RMSprop<T, V, VD> {
    pub fn new(lr: T, decay_rate: T) -> Self {
        Self {
            lr,
            decay_rate,
            h: None,
        }
    }
}

impl<SL, SR, DL, DR, E> Optimizer<SL, SR, DL, DR> for RMSprop<E, E, DL>
where
    SL: DataMut<Elem = E>,
    SR: Data<Elem = E>,
    E: Clone
        + std::fmt::Debug
        + Zero
        + AddAssign<E>
        + Mul<E, Output = E>
        + MulAssign<E>
        + 'static
        + Float
        + Add<E, Output = E>
        + ScalarOperand
        + SubAssign<E>,
    DL: Dimension,
    DR: Dimension + DimMax<DL>,
{
    fn step(&mut self, param: &mut ArrayBase<SL, DL>, grad: &ArrayBase<SR, DR>) {
        // 初始化v
        if self.h.is_none() {
            let default_value = Array::zeros::<DL>(param.raw_dim());

            self.h = Some(default_value);
        }

        let h_value = self.h.as_mut().unwrap();

        *h_value *= self.decay_rate;
        *h_value += &(grad * grad * (E::one() - self.decay_rate));

        let g_value = grad * self.lr.clone();
        let h_value = h_value.sqrt() + E::epsilon();

        let new = &g_value / &h_value;
        *param -= &new;
    }
}

pub struct Adam<T, V: std::fmt::Debug, DL: Dimension> {
    lr: T,
    beta1: T,
    beta2: T,
    iter: i32,
    m: Option<Array<V, DL>>,
    v: Option<Array<V, DL>>,
}

impl<SL, SR, E, DL, DR> Optimizer<SL, SR, DL, DR> for Adam<E, E, DL>
where
    SL: DataMut<Elem = E>,
    SR: Data<Elem = E>,
    E: Clone
        + std::fmt::Debug
        + Zero
        + Float
        + ScalarOperand
        + Mul<E, Output = E>
        + AddAssign<E>
        + SubAssign<E>,
    DL: Dimension,
    DR: Dimension + DimMax<DL, Output = DL>,
{
    fn step(&mut self, param: &mut ArrayBase<SL, DL>, grad: &ArrayBase<SR, DR>) {
        if self.m.is_none() {
            let default_value = Array::zeros::<DL>(param.raw_dim());

            self.m = Some(default_value.clone());
            self.v = Some(default_value);
        }

        self.iter += 1;

        let lr_t = self.lr * (E::one() - self.beta2.powi(self.iter))
            / (E::one() - self.beta1.powi(self.iter));

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        let m_value = (grad - &*m) * (E::one() - self.beta1);
        *m += &m_value;
        let v_value = (grad.pow2() - &*v) * (E::one() - self.beta2);
        *v += &v_value;

        let v1 = &*m * lr_t;
        let v2 = v.sqrt() - E::epsilon();
        *param -= &(&v1 / &v2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mat;

    #[test]
    fn test_sgd_update() {
        let mut sgd = Sgd { lr: 0.1 };

        let mut param = Mat::from_elem([2, 2], 1.0);
        let grad = Mat::from_elem([2, 2], 0.5);
        sgd.step(&mut param, &grad);
        assert_eq!(param, Mat::from_elem([2, 2], 0.95));

        {
            let mut sgd = Sgd { lr: 1 };

            let mut param = Mat::from_elem([2, 2], 5);
            let grad = Mat::from_elem([2, 2], 2);
            sgd.step(&mut param, &grad);
            println!("updated param: {param}");
            assert_eq!(param, Mat::from_elem([2, 2], 3));
        }

        {
            let mut sgd = Sgd { lr: 1 };

            let mut param = Mat::from_elem([2, 2], 5);
            let grad = Mat::from_elem([1, 1], 2);
            sgd.step(&mut param, &grad);
            println!("updated param: {param}");
            assert_eq!(param, Mat::from_elem([2, 2], 3));
        }
    }
}
