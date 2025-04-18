use std::fmt::Debug;

use ndarray::{Array, ArrayView, Dimension, NdIndex};
use num::Zero;

#[derive(Debug, Default)]
pub struct ReluLayer<D: Dimension> {
    mask: Option<Array<bool, D>>,
}

impl<D: Dimension> ReluLayer<D> {
    pub fn forward<T: Clone + Zero + PartialOrd>(&mut self, x: &ArrayView<T, D>) -> Array<T, D> {
        let zero = T::zero();

        self.mask = Some(x.mapv(|a| a <= zero.clone()));

        let out = x.mapv(|a| if a > zero.clone() { a } else { zero.clone() });

        out
    }

    pub fn backward<T: Clone + Zero>(&self, dout: &ArrayView<T, D>) -> Array<T, D>
    where
        D::Pattern: NdIndex<D>,
    {
        let mut out = dout.to_owned();

        let zero = T::zero();

        for (idx, a) in out.indexed_iter_mut() {
            let v = self.mask.as_ref().unwrap()[idx];

            if v {
                *a = zero.clone();
            }
        }

        out
    }
}
