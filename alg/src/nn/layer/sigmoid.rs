use num::Float;
use vectra::{NumExt, prelude::Array};

use crate::{math::ActivationFn, nn::layer::LayerWard};

#[derive(Debug)]
pub struct Sigmoid<const D: usize, T: Float + NumExt> {
    out: Option<Array<D, T>>,
}

impl<const D: usize, T: Float + NumExt> LayerWard<D, D, T> for Sigmoid<D, T> {
    fn forward(&mut self, input: Array<D, T>) -> Array<D, T> {
        let out = input.sigmoid();

        self.out = Some(out.clone());

        out
    }

    fn backward(&mut self, mut grad: Array<D, T>) -> Array<D, T> {
        grad.multi_iter_mut(|idx, item| {
            let y = self.out.as_ref().unwrap()[idx.map(|a| a as isize)];

            *item = *item * (T::one() - y) * y;
        });

        grad
    }
}
