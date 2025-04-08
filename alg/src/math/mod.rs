use ndarray::{Array, Dimension};
pub mod autodiff;
pub mod loss;
pub mod normalize;
pub mod one_hot;

pub struct DigitalRecognition;

pub trait Softmax {
    type Output;
    fn softmax(&self) -> Self::Output;
}

impl<D: Dimension> Softmax for Array<f64, D> {
    type Output = Array<f64, D>;

    fn softmax(&self) -> Self::Output {
        let max = self.iter().max_by(|a, b| (*a).total_cmp(*b)).unwrap();

        let exp_a = (self - *max).exp();

        let sum = exp_a.sum();

        exp_a / sum
    }
}

impl<D: Dimension> Softmax for Array<f32, D> {
    type Output = Array<f32, D>;

    fn softmax(&self) -> Self::Output {
        let max = self.iter().max_by(|a, b| (*a).total_cmp(*b)).unwrap();

        let exp_a = (self - *max).exp();

        let sum = exp_a.sum();

        exp_a / sum
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

        for (a, b) in dest.into_iter().zip(sm.into_iter()) {
            println!("{}", ((a - b) as f64).abs());
            assert_relative_eq!(a, b, max_relative = 0.000001);
        }
    }

    #[test]
    fn test_digital_recognition() {
        let a = array![12.0, 12.0, 2.0, 5.0];

        let b = DigitalRecognition::normalize(&a);
        println!("b= {b}");
    }
}
