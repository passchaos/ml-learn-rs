use std::ops::Mul;

use ndarray::{Array, Array1, Array2, ArrayBase, ArrayView1, Axis, Data, Dimension, NdFloat};
use num::traits::float::TotalOrder;
pub mod autodiff;
pub mod loss;
pub mod normalize;
pub mod one_hot;
pub mod stat;

pub struct DigitalRecognition;

pub trait Max {
    type Output;
    fn max_val(&self) -> &Self::Output;
}

impl<T: TotalOrder, S: Data<Elem = T>, D: Dimension> Max for ArrayBase<S, D> {
    type Output = T;

    fn max_val(&self) -> &Self::Output {
        self.iter().max_by(|a, b| a.total_cmp(b)).unwrap()
    }
}

pub trait Softmax {
    type Output;
    fn softmax(&self) -> Self::Output;
}

impl<T: NdFloat + TotalOrder> Softmax for Array1<T> {
    type Output = Array1<T>;

    fn softmax(&self) -> Self::Output {
        let max = self.max_val();

        let exp_a = (self - *max).exp();

        let sum = exp_a.sum();

        exp_a / sum
    }
}

impl<T: NdFloat + TotalOrder> Softmax for Array2<T> {
    type Output = Array2<T>;

    fn softmax(&self) -> Self::Output {
        // let max = self.max_val();

        // let exp_a = (self - *max).exp();

        // let sum = exp_a.sum();

        // exp_a / sum
        let max = self.map_axis(Axis(1), |a| *(a.max_val()));

        let mut exp_a = self.clone();
        for (mut a, b) in exp_a.axis_iter_mut(Axis(0)).zip(max.into_iter()) {
            a.map_inplace(|x| *x = (*x - b).exp());

            let sum = a.sum();

            a.map_inplace(|x| *x /= sum);
        }

        exp_a
    }
}

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

impl<D: Dimension> Sigmoid for Array<f32, D> {
    type Output = Array<f32, D>;

    fn sigmoid(&self) -> Self::Output {
        1.0 / (1.0 + (-self).exp())
    }
}

impl Sigmoid for f32 {
    type Output = f32;

    fn sigmoid(&self) -> Self::Output {
        1.0 / (1.0 + (-self).exp())
    }
}

pub trait Relu {
    type Output;
    fn relu(&self) -> Self::Output;
}

impl<D: Dimension> Relu for Array<f64, D> {
    type Output = Array<f64, D>;

    fn relu(&self) -> Self::Output {
        let mut data = self.clone();

        data.map_inplace(|a| {
            if *a < 0.0 {
                *a = 0.0;
            }
        });

        data
    }
}

impl Relu for f64 {
    type Output = f64;

    fn relu(&self) -> Self::Output {
        if self > &0.0 { *self } else { 0.0 }
    }
}

impl<D: Dimension> Relu for Array<f32, D> {
    type Output = Array<f32, D>;

    fn relu(&self) -> Self::Output {
        let mut data = self.clone();

        data.map_inplace(|a| {
            if *a < 0.0 {
                *a = 0.0;
            }
        });

        data
    }
}

impl Relu for f32 {
    type Output = f32;

    fn relu(&self) -> Self::Output {
        if self > &0.0 { *self } else { 0.0 }
    }
}

pub trait L2Norm<Output> {
    fn l2_norm(&self) -> Output;
}

impl<'a, T: NdFloat> L2Norm<T> for ArrayView1<'a, T> {
    fn l2_norm(&self) -> T {
        self.pow2().sum().sqrt()
    }
}

pub fn cos_similarity<T: NdFloat>(x: &ArrayView1<T>, y: &ArrayView1<T>) -> T {
    let x_sum_sq = x.l2_norm() + T::epsilon();
    let y_sum_sq = y.l2_norm() + T::epsilon();

    let nx = x / x_sum_sq;
    let ny = y / y_sum_sq;

    let value = nx.dot(&ny);
    value
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::array;

    use super::{normalize::NormalizeTransform, *};

    #[test]
    fn test_softmax() {
        let input = array![0.3, 2.9, 4.0];

        let sm: Array<_, _> = input.softmax();
        let dest = array![0.01821127, 0.24519181, 0.73659691];

        assert_relative_eq!(sm, dest, max_relative = 0.000001);

        let a = array![
            [0.1, 0.05, 0.6, 0., 0.05, 0.1, 0., 0.1, 0., 0.],
            [0.1, 0.15, 0.5, 0., 0.05, 0.1, 0., 0.1, 0., 0.]
        ];

        let sv = a.softmax();
        assert_relative_eq!(
            sv,
            array![
                [
                    0.09832329, 0.09352801, 0.16210771, 0.0889666, 0.09352801, 0.09832329,
                    0.0889666, 0.09832329, 0.0889666, 0.0889666
                ],
                [
                    0.09887603, 0.10394551, 0.1475057, 0.08946673, 0.09405379, 0.09887603,
                    0.08946673, 0.09887603, 0.08946673, 0.08946673
                ]
            ],
            max_relative = 0.000001
        );
    }

    #[test]
    fn test_digital_recognition() {
        let a = array![12.0, 12.0, 2.0, 5.0];

        let b = DigitalRecognition::normalize(&a);
        println!("b= {b}");
    }
}
