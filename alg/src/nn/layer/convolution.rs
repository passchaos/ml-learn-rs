use itertools::Itertools;
use vectra::{NumExt, prelude::Array};

use crate::nn::layer::LayerWard;

struct ConvolutionLayer<T> {
    weight: Array<4, T>,
    bias: Array<1, T>,
    stride: usize,
    pad: usize,
}

impl<T> ConvolutionLayer<T> {
    fn new(weight: Array<4, T>, bias: Array<1, T>, stride: usize, pad: usize) -> Self {
        assert_eq!(weight.shape()[0], bias.shape()[0]);

        ConvolutionLayer {
            weight,
            bias,
            stride,
            pad,
        }
    }
}

impl<T> LayerWard<4, 2, T> for ConvolutionLayer<T>
where
    T: NumExt,
{
    fn forward(&mut self, input: Array<4, T>) -> Array<2, T> {
        // let [f_n, c, f_h, f_w] = self.weight.shape();
        // let [n, _, h, w] = input.shape();

        // let out = conv3(input, self.weight, pad, stride);

        // out
        todo!()
    }

    fn backward(&mut self, grad: Array<2, T>) -> Array<4, T> {
        todo!()
    }
}

fn conv3<T>(a: &Array<4, T>, b: &Array<4, T>, pad: usize, stride: usize) -> Array<4, T>
where
    T: NumExt,
{
    let a = a.pad((pad, pad, pad, pad), T::zero());

    let [i_n, i_c, i_h, i_w] = a.shape();

    let [w_n, _, w_h, w_w] = b.shape();

    let o_h = (i_h - w_h) / stride + 1;
    let o_w = (i_w - w_w) / stride + 1;

    let indices = (0..w_h).cartesian_product(0..w_w);

    let mut result_data = vec![T::zero(); i_n * w_n * o_h * o_w];

    for ni in 0..i_n {
        for ci in 0..i_c {
            for i in 0..o_h {
                for j in 0..o_w {
                    for k in 0..w_n {
                        for index in indices.clone() {
                            let (ii, jj) = index;

                            let a_idx =
                                [ni, ci, ii + stride * i, jj + stride * j].map(|i| i as isize);
                            let b_idx = [k, ci, ii, jj].map(|i| i as isize);

                            let a_val = a[a_idx];
                            let b_val = b[b_idx];
                            result_data[ni * w_n * o_h * o_w + k * o_h * o_w + i * o_w + j] +=
                                a_val * b_val;
                        }
                    }
                }
            }
        }
    }

    Array::from_vec(result_data, [i_n, w_n, o_h, o_w])
}

fn conv<T>(a: &Array<2, T>, b: &Array<2, T>, pad: usize, stride: usize) -> Array<2, T>
where
    T: NumExt,
{
    let a = a.pad((pad, pad, pad, pad), T::zero());
    let s_h = a.shape()[0];
    let s_w = a.shape()[1];
    let o_h = b.shape()[0];
    let o_w = b.shape()[1];

    let o_h = (s_h - o_h) / stride + 1;
    let o_w = (s_w - o_w) / stride + 1;

    let mut result_data = vec![T::zero(); o_h * o_w];

    for i in 0..o_h {
        for j in 0..o_w {
            let v = b
                .multi_iter()
                .map(|(idx, v)| {
                    let a_idx = [i * stride + idx[0], j * stride + idx[1]].map(|a| a as isize);
                    let a_val = a[a_idx];
                    a_val * *v
                })
                .sum();

            result_data[i * o_w + j] = v;
        }
    }

    Array::from_vec(result_data, [o_h, o_w])
}

fn pool<'a, T, F>(input: &'a Array<4, T>, stride: usize, f: F) -> Array<4, T>
where
    T: NumExt + 'static,
    F: Fn(Vec<([isize; 4], &'a T)>) -> T,
{
    let i_shape = input.shape();

    let [i_n, i_c, i_h, i_w] = i_shape;

    let o_h = (i_h - stride) / stride + 1;
    let o_w = (i_w - stride) / stride + 1;

    let mut result_data = vec![T::zero(); i_n * i_c * o_h * o_w];

    let car_indices = (0..o_h).cartesian_product(0..o_w);

    for n in 0..i_n {
        for c in 0..i_c {
            for i in 0..o_h {
                for j in 0..o_w {
                    let values: Vec<_> = car_indices
                        .clone()
                        .map(|(i_h, i_w)| {
                            let a_idx = [n, c, i * stride + i_h, j * stride + i_w];

                            let a_idx = a_idx.map(|i| i as isize);

                            let v = &input[a_idx];

                            (a_idx, v)
                        })
                        .collect();

                    let v = f(values);

                    result_data[n * i_c * o_h * o_w + c * o_h * o_w + i * o_w + j] = v;
                }
            }
        }
    }

    Array::from_vec(result_data, [i_n, i_c, o_h, o_w])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learn() {
        let a = Array::from_vec(vec![1, 2, 3, 0, 0, 1, 2, 3, 3, 0, 1, 2, 2, 3, 0, 1], [4, 4]);
        let b = Array::from_vec(vec![2, 0, 1, 0, 1, 2, 1, 0, 2], [3, 3]);

        let c = conv(&a, &b, 0, 1);
        assert_eq!(c, Array::from_vec(vec![15, 16, 6, 15], [2, 2]));

        let c = &c + &Array::from_vec(vec![3], [1, 1]);
        assert_eq!(c, Array::from_vec(vec![18, 19, 9, 18], [2, 2]));
        println!("c: {c:?}");

        let c1 = conv(&a, &b, 1, 1);
        println!("c1: {c1:?}");

        let a = Array::from_vec(
            vec![1, 2, 3, 0, 0, 1, 2, 3, 3, 0, 1, 2, 2, 3, 0, 1],
            [1, 1, 4, 4],
        );
        let b = Array::from_vec(vec![2, 0, 1, 0, 1, 2, 1, 0, 2], [1, 1, 3, 3]);

        let c = conv3(&a, &b, 0, 1);
        assert_eq!(
            c,
            Array::from_vec(vec![15, 16, 6, 15], [2, 2])
                .unsqueeze(0)
                .unsqueeze(0)
        );

        let c = &c + &Array::from_vec(vec![3], [1, 1, 1, 1]);
        assert_eq!(c, Array::from_vec(vec![18, 19, 9, 18], [1, 1, 2, 2]));
        println!("c: {c:?}");

        let c1 = conv3(&a, &b, 1, 1);
        println!("c1: {c1:?}");

        let a = Array::from_vec(
            vec![1, 2, 3, 0, 0, 1, 2, 3, 3, 0, 1, 2, 2, 3, 0, 1],
            [1, 1, 4, 4],
        );
        let a = Array::cat(&vec![&a, &a], 1);
        let a = Array::cat(&vec![&a, &a], 0);
        let b = Array::from_vec(vec![2, 0, 1, 0, 1, 2, 1, 0, 2], [1, 1, 3, 3]);
        let b = Array::cat(&vec![&b, &b], 1);
        let b = Array::cat(&vec![&b, &b], 0);

        println!("a: {a:?} b: {b:?}");
        let c = conv3(&a, &b, 0, 1);

        let res = Array::from_vec(vec![15, 16, 6, 15], [1, 1, 2, 2]).mul_scalar(2);

        let res = Array::cat(&[&res, &res], 1);
        assert_eq!(c, Array::cat(&[&res, &res], 0));
    }

    #[test]
    fn test_pool() {
        let a = Array::from_vec(
            vec![1, 2, 1, 0, 0, 1, 2, 3, 3, 0, 1, 2, 2, 4, 0, 1],
            [1, 1, 4, 4],
        );
        let a = Array::cat(&[&a, &a], 1);
        let b = pool(&a, 2, |values| {
            let v = values.into_iter().map(|(_i, v)| v).max().unwrap();

            *v
        });

        assert_eq!(
            b,
            Array::from_vec(vec![2, 3, 4, 2, 2, 3, 4, 2], [1, 2, 2, 2])
        );
        println!("pooled value: {b:?}");
    }
}
