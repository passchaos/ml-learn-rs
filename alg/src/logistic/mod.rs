use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

pub fn gradient_ascent(data_in: ArrayView2<f64>, labels_in: ArrayView1<f64>) -> Array1<f64> {
    let m = data_in.shape()[0];
    let n = data_in.shape()[1];
    let alpha = 0.001;
    let max_cycles = 500;

    let labels_in = Array2::from_shape_vec((m, 1), labels_in.to_vec()).unwrap();

    let mut weights: Array2<f64> = Array2::ones((n, 1));

    for _ in 0..max_cycles {
        let h = crate::math::sigmoid(data_in.dot(&weights));

        let error = &labels_in - &h;
        weights = weights + alpha * data_in.t().dot(&error);
    }

    weights.index_axis_move(Axis(1), 0)
}

pub fn stoc_grad_ascent_0(data_in: ArrayView2<f64>, labels_in: ArrayView1<f64>) -> Array1<f64> {
    let m = data_in.shape()[0];
    let n = data_in.shape()[1];

    let alpha = 0.01;

    let mut weights = Array2::ones((n, 1));

    for i in 0..m {
        let data_i_orig = data_in.index_axis(Axis(0), i);
        let data_i = Array2::from_shape_vec((1, n), data_i_orig.clone().to_vec()).unwrap();

        let h = crate::math::sigmoid(data_i.dot(&weights));

        let h = h.first().unwrap();
        let error = labels_in[i] - h;

        weights = weights + alpha * error * data_i.t().to_owned();
    }

    weights.index_axis_move(Axis(1), 0)
}
