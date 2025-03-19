use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::{distributions::Uniform, prelude::Distribution};

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

pub fn stoc_grad_ascent_0(
    weight_iterations: &mut Vec<Array1<f64>>,
    data_in: ArrayView2<f64>,
    labels_in: ArrayView1<f64>,
    num_iter: usize,
) -> Array1<f64> {
    let m = data_in.shape()[0];
    let n = data_in.shape()[1];

    let alpha = 0.01;

    let mut weights = Array2::ones((n, 1));

    for _ in 0..num_iter {
        for i in 0..m {
            let data_i_orig = data_in.index_axis(Axis(0), i);
            let data_i = Array2::from_shape_vec((1, n), data_i_orig.clone().to_vec()).unwrap();

            let h = crate::math::sigmoid(data_i.dot(&weights));

            let h = h.first().unwrap();
            let error = labels_in[i] - h;

            weights = weights + alpha * error * data_i.t().to_owned();
        }

        let weight_i = weights.index_axis(Axis(1), 0).to_owned();
        weight_iterations.push(weight_i);
    }

    weights.index_axis_move(Axis(1), 0)
}

pub fn stoc_grad_ascent_1(
    weight_iterations: &mut Vec<Array1<f64>>,
    data_in: ArrayView2<f64>,
    labels_in: ArrayView1<f64>,
    num_iter: usize,
) -> Array1<f64> {
    let m = data_in.shape()[0];
    let n = data_in.shape()[1];

    let mut weights = Array2::ones((n, 1));

    for j in 0..num_iter {
        let mut data_index: Vec<_> = (0..m).collect();

        for i in 0..m {
            let alpha = 4.0 / (1.0 + j as f64 + i as f64) + 0.0001;

            let between = Uniform::from(0..data_index.len());
            let mut rng = rand::thread_rng();

            let rand_index = between.sample(&mut rng);

            let data_i_orig = data_in.index_axis(Axis(0), rand_index);
            let data_i = Array2::from_shape_vec((1, n), data_i_orig.clone().to_vec()).unwrap();

            let data_sum = data_i.dot(&weights);
            let h = crate::math::sigmoid(data_sum.clone());

            let h = h.first().unwrap();
            let error = labels_in[rand_index] - h;

            weights = weights + alpha * error * data_i.t().to_owned();

            data_index.remove(rand_index);
        }

        let weight_i = weights.index_axis(Axis(1), 0).to_owned();
        weight_iterations.push(weight_i);
    }

    weights.index_axis_move(Axis(1), 0)
}
