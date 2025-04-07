use ndarray::{Array, Dimension};

use super::DigitalRecognition;

pub trait NormalizeTransform<'a> {
    fn normalize<D: Clone + Into<f32> + 'static, A: Dimension>(
        input: &'a Array<D, A>,
    ) -> Array<f32, A>;
}

impl<'a> NormalizeTransform<'a> for DigitalRecognition {
    fn normalize<D: Clone + Into<f32> + 'static, A: Dimension>(
        input: &'a Array<D, A>,
    ) -> Array<f32, A> {
        input.mapv(|a| a.into() / 255.0)
    }
}
