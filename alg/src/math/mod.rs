use ndarray::{Array, Dimension};

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
