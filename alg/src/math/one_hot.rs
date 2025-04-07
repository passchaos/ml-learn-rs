use ndarray::{Array1, Array2};

use super::DigitalRecognition;

pub trait OneHotTransform<'a> {
    type InputD;
    type OutputD;

    fn one_hot(input: &'a Self::InputD) -> Self::OutputD;
}

impl<'a> OneHotTransform<'a> for DigitalRecognition {
    type InputD = Array1<u8>;
    type OutputD = Array2<f32>;

    fn one_hot(input: &'a Self::InputD) -> Self::OutputD {
        let mut data = Array2::zeros((input.len(), 10));

        for (row, value) in input.iter().enumerate() {
            data[(row, *value as usize)] = 1.0;
        }

        data
    }
}
