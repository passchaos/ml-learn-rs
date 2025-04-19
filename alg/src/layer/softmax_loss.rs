use ndarray::{Array, Array2, Dimension, NdFloat};

use crate::math::{Softmax, loss::cross_entropy_error};

#[derive(Default)]
pub struct SoftmaxWithLossLayer<T, D> {
    y: Option<Array<T, D>>,
    t: Option<Array<T, D>>,
}

impl<T: Clone + NdFloat + From<f32>, D: Dimension> SoftmaxWithLossLayer<T, D> {
    pub fn forward(&mut self, x: &Array<T, D>, t: &Array<T, D>) -> T
    where
        Array<T, D>: Softmax<Output = Array<T, D>>,
    {
        let y = x.softmax();
        self.y = Some(y.clone());

        cross_entropy_error(y, t.to_owned())
    }

    pub fn backward(&self) -> Array<T, D> {
        let batch_size: T = (self.t.as_ref().unwrap().shape()[0] as f32).into();

        let dx = (self.y.as_ref().unwrap() - self.t.as_ref().unwrap()) / batch_size;

        dx
    }
}
