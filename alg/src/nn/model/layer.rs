use std::marker::PhantomData;

use num::Float;
use vectra::{NumExt, prelude::Matmul};

use crate::{
    math::{ActivationFn, LossFn},
    nn::matmul_policy,
};

pub type Mat<T> = vectra::prelude::Array<2, T>;
pub type MatN<T> = ndarray::Array2<T>;

#[derive(Debug, Default)]
pub struct Relu<T> {
    mask: Option<Mat<bool>>,
    phantom: PhantomData<T>,
}

impl<T: NumExt> Relu<T> {
    pub fn forward(&mut self, mut x: Mat<T>) -> Mat<T> {
        let mask = x.map(|&a| a.cmp_ext(&T::zero()).is_le());

        x.mask_fill(&mask, T::zero());

        self.mask = Some(mask);

        x
    }

    pub fn backward(&self, mut grad: Mat<T>) -> Mat<T> {
        grad.mask_fill(self.mask.as_ref().unwrap(), T::zero());

        grad
    }
}

#[derive(Default)]
pub struct Sigmoid<T> {
    output: Option<Mat<T>>,
}

impl<T: NumExt + Float> Sigmoid<T> {
    pub fn forward(&mut self, x: Mat<T>) -> Mat<T> {
        let output = x.sigmoid();

        self.output = Some(output.clone());

        output
    }

    pub fn backward(&self, mut grad: Mat<T>) -> Mat<T> {
        let output = self.output.as_ref().unwrap();

        grad *= &(&((-output.clone()).add_scalar(T::one())) * output);

        grad
    }
}

pub struct AffineN<T> {
    pub w: MatN<T>,
    pub b: MatN<T>,
    x: Option<MatN<T>>,
    pub dw: Option<MatN<T>>,
    pub db: Option<MatN<T>>,
}

impl<T> AffineN<T> {
    pub fn new(w: MatN<T>, b: MatN<T>) -> Self {
        Self {
            w,
            b,
            x: None,
            dw: None,
            db: None,
        }
    }
}

impl<T: NumExt + Float + 'static> AffineN<T> {
    pub fn forward(&mut self, x: MatN<T>) -> MatN<T> {
        let output = x.dot(&self.w);

        self.x = Some(x);

        output
    }

    pub fn backward(&mut self, grad: MatN<T>) -> MatN<T> {
        let output = grad.dot(&self.w.t());

        let dw = self.x.as_ref().unwrap().t().dot(&grad);

        self.dw = Some(dw);
        // self.db = Some(grad.sum_axis(0));

        output
    }
}

pub struct Affine<T> {
    pub w: Mat<T>,
    pub b: Mat<T>,
    x: Option<Mat<T>>,
    pub dw: Option<Mat<T>>,
    pub db: Option<Mat<T>>,
}

impl<T> Affine<T> {
    pub fn new(w: Mat<T>, b: Mat<T>) -> Self {
        Affine {
            w,
            b,
            x: None,
            dw: None,
            db: None,
        }
    }
}

impl<T: NumExt + Float> Affine<T>
where
    Mat<T>: Matmul,
{
    pub fn forward(&mut self, x: Mat<T>) -> Mat<T> {
        let output = x.matmul(&self.w, matmul_policy());
        self.x = Some(x);

        output
    }

    pub fn backward(&mut self, grad: Mat<T>) -> Mat<T> {
        let output = grad.matmul(&self.w.clone().transpose(), matmul_policy());

        let dw = self
            .x
            .clone()
            .unwrap()
            .transpose()
            .matmul(&grad, matmul_policy());

        self.dw = Some(dw);
        // self.db = Some(grad.sum_axis(0));

        output
    }
}

#[derive(Default)]
pub struct SoftmaxWithLoss<T> {
    loss: Option<Mat<T>>,
    y: Option<Mat<T>>,
    t: Option<Mat<T>>,
}

impl<T: NumExt + Float> SoftmaxWithLoss<T> {
    pub fn forward(&mut self, y: Mat<T>, t: Mat<T>) -> T {
        let y = y.softmax();

        let loss = y.cross_entropy_error(&t);

        self.y = Some(y);
        self.t = Some(t);

        loss
    }

    pub fn backward(&self) -> Mat<T> {
        let batch_size = self.y.as_ref().unwrap().shape()[0];

        let tmp = (self.y.as_ref().unwrap() - self.t.as_ref().unwrap())
            .div_scalar(T::from(batch_size).unwrap());

        tmp
    }
}

pub fn cross_entropy_error<T: NumExt + Float + 'static>(
    y: &ndarray::Array2<T>,
    t: &ndarray::Array2<T>,
) -> T {
    // let mut y = y.into_dyn();
    // let mut t = t.into_dyn();

    let delta = 1e-7;

    let batch_size = y.shape()[0] as f32;

    let y1 = y.mapv(|a| (a + T::from(delta).unwrap()));
    let y1 = y1.ln();

    let mut v1 = t * y1;
    v1.mapv_inplace(|a| -a);
    v1.sum() / T::from(batch_size).unwrap()
}

fn max_value<T: NumExt>(x: ndarray::ArrayBase<ndarray::ViewRepr<&T>, ndarray::Ix1>) -> T {
    x.into_iter().max_by(|&a, &b| a.cmp_ext(b)).unwrap().clone()
}

fn softmax<T: NumExt + Float + 'static>(x: &MatN<T>) -> MatN<T> {
    let max = x
        .map_axis(ndarray::Axis(1), |a| max_value(a))
        .insert_axis(ndarray::Axis(1));

    let exp_a = x.clone();

    let a = (exp_a - max).exp();
    let a_t = a.sum_axis(ndarray::Axis(1)).insert_axis(ndarray::Axis(1));

    a / a_t
}

#[derive(Default)]
pub struct SoftmaxWithLossN<T> {
    loss: Option<MatN<T>>,
    y: Option<MatN<T>>,
    t: Option<MatN<T>>,
}

impl<T: NumExt + Float + 'static + ndarray::ScalarOperand> SoftmaxWithLossN<T> {
    pub fn forward(&mut self, y: MatN<T>, t: MatN<T>) -> T {
        let y = softmax(&y);

        let loss = cross_entropy_error(&y, &t);

        self.y = Some(y);
        self.t = Some(t);

        loss
    }

    pub fn backward(&self) -> MatN<T> {
        let batch_size = self.y.as_ref().unwrap().shape()[0];

        let tmp =
            (self.y.as_ref().unwrap() - self.t.as_ref().unwrap()) / T::from(batch_size).unwrap();

        tmp
    }
}
