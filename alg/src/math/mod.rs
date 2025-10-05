use num::{Float, cast};
use vectra::{NumExt, prelude::Array};

use crate::nn::delta;

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

impl<const D: usize, T: Float + NumExt> ActivationFn for Array<D, T> {
    fn sigmoid(&self) -> Self {
        (-self.clone()).exp().add_scalar(T::one()).recip()
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

impl<T: Float + NumExt> LossFn for Array<1, T> {
    type Output = T;

    fn mean_squared_error(&self, y: &Self) -> Self::Output {
        (self - y).pow2().sum() / (T::one() + T::one())
    }

    fn cross_entropy_error(&self, y: &Self) -> Self::Output {
        let log = self.map(|&x| x + delta()).ln();

        (&log * y).into_map(|a| -a).sum()
    }
}

impl<T: Float + NumExt> LossFn for Array<2, T> {
    type Output = T;

    fn mean_squared_error(&self, t: &Self) -> Self::Output {
        (self - t).pow2().sum() / (T::one() + T::one())
    }

    fn cross_entropy_error(&self, t: &Self) -> Self::Output {
        let batch_size = self.shape()[0];

        let y = self.map(|&x| x + delta()).ln();

        (t * &y).into_map(|a| -a).sum() / cast::<_, _>(batch_size).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use vectra::prelude::Array;

    use crate::math::{ActivationFn, LossFn};

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

    #[test]
    fn test_cross_entropy_error() {
        let t = Array::<_, f32>::from(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let y = Array::from(vec![0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]);

        let err1 = y.mean_squared_error(&t);
        let err2 = y.cross_entropy_error(&t);

        println!("mse= {err1} cee= {err2}");
        assert_relative_eq!(err1, 0.097500000000000031);
        assert_relative_eq!(err2, 0.510825457, epsilon = 1e-6);

        let y = Array::<_, f32>::from_vec(
            vec![
                0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0, 0.01, 0.15, 0.06, 0.55, 0.03,
                0.07, 0.08, 0.01, 0.01, 0.03,
            ],
            [2, 10],
        );
        let t = Array::from_vec(
            vec![
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
            ],
            [2, 10],
        );

        let err1 = y.mean_squared_error(&t);
        let err2 = y.cross_entropy_error(&t);

        println!("mse= {err1} cee= {err2}");
        assert_relative_eq!(err1, 0.21849997, epsilon = 1e-7);
        assert_relative_eq!(err2, 0.55433106, epsilon = 1e-7);
    }
}
