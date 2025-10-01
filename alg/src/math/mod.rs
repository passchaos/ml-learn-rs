use std::ops::{AddAssign, MulAssign};

use num::{Float, pow::Pow, traits::float::TotalOrder};
use vectra::prelude::Array;
pub mod autodiff;
pub mod loss;

pub trait ActivationFn {
    fn softmax(&self) -> Self;
    fn sigmoid(&self) -> Self;
    fn relu(&self) -> Self;
}

pub trait LossFn {
    type Output;
    fn mean_squared_error(&self, y: &Self) -> Self::Output;
    fn cross_entropy_error(&self, y: &Self) -> Self::Output;
}

impl<const D: usize, T: Float + TotalOrder + Default + AddAssign + MulAssign + Pow<T, Output = T>>
    ActivationFn for Array<D, T>
{
    fn sigmoid(&self) -> Self {
        self.clone()
            .mul_scalar(-T::one())
            .exp()
            .add_scalar(T::one())
            .pow(-T::one())
    }

    fn relu(&self) -> Self {
        self.map(|x| x.max(T::zero()))
    }

    fn softmax(&self) -> Self {
        let a = self.max_axis((D - 1) as isize);
        let a = (self - &a).exp();
        let a_t = a.sum_axis((D - 1) as isize);

        &a / &a_t
    }
}

impl<const D: usize, T: Float + Default> LossFn for Array<D, T> {
    type Output = T;

    fn mean_squared_error(&self, y: &Self) -> Self::Output {
        (self - y).pow2().sum() / (T::one() + T::one())
    }

    fn cross_entropy_error(&self, y: &Self) -> Self::Output {
        let eps = T::epsilon();
        let log = self.map(|x| x.max(eps).ln());

        (&log * y).sum_axis(-1).mean().neg()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use vectra::prelude::Array;

    use crate::math::ActivationFn;

    #[test]
    fn test_softmax() {
        let input = Array::from(vec![0.3, 2.9, 4.0]);

        let sm: Array<_, _> = input.softmax();
        let dest = Array::from(vec![0.01821127, 0.24519181, 0.73659691]);

        assert_relative_eq!(sm, dest, max_relative = 0.000001);

        let a = Array::from_vec(
            vec![
                0.1, 0.05, 0.6, 0., 0.05, 0.1, 0., 0.1, 0., 0., 0.1, 0.15, 0.5, 0., 0.05, 0.1, 0.,
                0.1, 0., 0.,
            ],
            [2, 10],
        );

        let sv = a.softmax();
        assert_relative_eq!(
            sv,
            Array::from_vec(
                vec![
                    0.09832329, 0.09352801, 0.16210771, 0.0889666, 0.09352801, 0.09832329,
                    0.0889666, 0.09832329, 0.0889666, 0.0889666, 0.09887603, 0.10394551, 0.1475057,
                    0.08946673, 0.09405379, 0.09887603, 0.08946673, 0.09887603, 0.08946673,
                    0.08946673
                ],
                [2, 10]
            ),
            max_relative = 0.000001
        );
    }
}
