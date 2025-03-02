use ndarray::Array2;

pub fn sigmoid(x: Array2<f64>) -> Array2<f64> {
    1.0 / (1.0 + (-x).exp())
}
