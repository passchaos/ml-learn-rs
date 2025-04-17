use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use rand::{distr::Uniform, prelude::Distribution};

use crate::math::Sigmoid;

pub fn gradient_ascent(
    data_in: ArrayView2<f64>,
    labels_in: ArrayView1<f64>,
    num_iter: usize,
) -> Array1<f64> {
    let n = data_in.shape()[1];
    let alpha = 0.001;

    let mut weights: Array1<f64> = Array1::ones(n);

    for _ in 0..num_iter {
        let h = data_in.dot(&weights).sigmoid();

        let error = &labels_in - &h;
        weights = weights + alpha * data_in.t().dot(&error);
    }

    weights
}

pub fn stoc_grad_ascent_0(
    data_in: ArrayView2<f64>,
    labels_in: ArrayView1<f64>,
    num_iter: usize,
) -> (Array1<f64>, Vec<Array1<f64>>) {
    let mut weight_iterations = vec![];

    let m = data_in.shape()[0];
    let n = data_in.shape()[1];

    let alpha = 0.01;

    let mut weights = Array1::ones(n);

    for _ in 0..num_iter {
        for i in 0..m {
            let data_i_orig = data_in.index_axis(Axis(0), i);

            let h = data_i_orig.dot(&weights).sigmoid();

            let error = labels_in[i] - h;

            weights = weights + alpha * error * data_i_orig.to_owned();
        }

        weight_iterations.push(weights.clone());
    }

    (weights, weight_iterations)
}

pub fn stoc_grad_ascent_1(
    data_in: ArrayView2<f64>,
    labels_in: ArrayView1<f64>,
    num_iter: usize,
) -> (Array1<f64>, Vec<Array1<f64>>) {
    let mut weight_iterations = vec![];

    let m = data_in.shape()[0];
    let n = data_in.shape()[1];

    let mut weights = Array1::ones(n);

    for j in 0..num_iter {
        let mut data_index: Vec<_> = (0..m).collect();

        for i in 0..m {
            let alpha = 4.0 / (1.0 + j as f64 + i as f64) + 0.0001;

            let between = Uniform::try_from(0..data_index.len()).unwrap();
            let mut rng = rand::rng();

            let rand_index = between.sample(&mut rng);

            let data_i_orig = data_in.index_axis(Axis(0), rand_index);

            let h = data_i_orig.dot(&weights).sigmoid();

            let error = labels_in[rand_index] - h;

            weights = weights + alpha * error * data_i_orig.to_owned();

            data_index.remove(rand_index);
        }

        weight_iterations.push(weights.clone());
    }

    (weights, weight_iterations)
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;

    #[test]
    fn test_dot_action() {
        let a = Array2::from_shape_vec((2, 4), vec![1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let b = Array1::from_vec(vec![4, 5, 6, 7]);

        println!("a: {a} b: {b}");

        println!(
            "dot= {} dot_t= {} chengji= {}",
            a.dot(&b),
            a.dot(&b.t()),
            a * b
        );
    }
}
