use ndarray::{Array, Dimension};

pub trait Sigmoid {
    type Output;
    fn sigmoid(&self) -> Self::Output;
}

impl<D: Dimension> Sigmoid for Array<f64, D> {
    type Output = Array<f64, D>;

    fn sigmoid(&self) -> Self::Output {
        1.0 / (1.0 + (-self).exp())
    }
}

impl Sigmoid for f64 {
    type Output = f64;

    fn sigmoid(&self) -> Self::Output {
        1.0 / (1.0 + (-self).exp())
    }
}
