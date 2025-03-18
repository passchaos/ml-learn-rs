use ndarray::{Array2, ArrayView2};

pub fn gradient_ascent(data_in: ArrayView2<f64>, labels_in: ArrayView2<f64>) -> Array2<f64> {
    let n = data_in.shape()[1];
    let alpha = 0.001;
    let max_cycles = 500;

    let mut weights: Array2<f64> = Array2::ones((n, 1));

    for _ in 0..max_cycles {
        let h = crate::math::sigmoid(data_in.dot(&weights));

        let error = &labels_in - &h;
        weights = weights + alpha * data_in.t().dot(&error);
    }

    weights
}
