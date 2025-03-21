use ndarray::{Array, Dimension};

pub fn sigmoid<Dim: Dimension>(x: &Array<f64, Dim>) -> Array<f64, Dim> {
    1.0 / (1.0 + (-x).exp())
}
