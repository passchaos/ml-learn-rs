use ndarray::{Array, ArrayView, Dimension, NdIndex};
use num::Float;

pub struct SigmoidLayer<T, D> {
    out: Option<Array<T, D>>,
}

impl<T: Float, D: Dimension> SigmoidLayer<T, D> {
    pub fn forward(&mut self, x: &ArrayView<T, D>) -> Array<T, D> {
        let out = x.mapv(|a| T::one() / (T::one() + (-a).exp()));

        self.out = Some(out.clone());

        out
    }

    pub fn backward(&self, dout: &ArrayView<T, D>) -> Array<T, D>
    where
        D::Pattern: NdIndex<D>,
    {
        let mut dx = dout.to_owned();

        for ((idx, item), dout_value) in dx.indexed_iter_mut().zip(dout.iter()) {
            let y = self.out.as_ref().unwrap()[idx];

            *item = *dout_value * (T::one() - y) * y;
        }

        dx
    }
}
