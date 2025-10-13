use std::any::type_name;

use num::Float;
use vectra::{NumExt, prelude::*};

type Ft = f32;

pub fn delta<T: Float>() -> T {
    match type_name::<T>() {
        "f32" => T::from(1.0e-7).unwrap(),
        "f64" => T::from(1.0e-15).unwrap(),
        _ => T::from(1.0e-7).unwrap(),
    }
}

pub fn print_mat_stat_info_ndarray(grad: &ndarray::Array2<f32>, str_prefix: &str) {
    let sum = grad.sum();
    let mean = grad.mean().unwrap();
    let var = grad.var(0.0);
    println!("{str_prefix} sum= {sum} mean= {mean} var= {var}");
}

pub fn print_mat_stat_info(grad: &Array<2, f32>, str_prefix: &str) {
    let sum = grad.sum();
    let mean = grad.mean::<f32>();
    let var = grad.var(0.0);
    println!("{str_prefix} sum= {sum} mean= {mean} var= {var}");
}

pub fn numerical_gradient<F, const D: usize, T: NumExt>(
    mut f: F,
    mut x: Array<D, T>,
) -> (Array<D, T>, Array<D, T>)
where
    F: FnMut(&Array<D, T>) -> T,
{
    let h = T::from(1e-4).unwrap();
    let mut grad = Array::zeros(x.shape());

    let indices: Vec<_> = x.multi_iter().map(|(idx, _)| idx).collect();

    for idx in indices {
        let idx = idx.map(|i| i as isize);

        let tmp_val = x[idx];

        x[idx] = tmp_val + h;
        let res1 = f(&x);

        x[idx] = tmp_val - h;
        let res2 = f(&x);

        grad[idx] = (res1 - res2) / (h + h);

        x[idx] = tmp_val;
    }

    (grad, x)
}

pub fn gradient_descent<F, const D: usize, T: NumExt>(
    mut f: F,
    mut init_x: Array<D, T>,
    lr: T,
    step_num: usize,
) -> Array<D, T>
where
    F: FnMut(&Array<D, T>) -> T,
{
    for _ in 0..step_num {
        let (grad, init_x_n) = numerical_gradient(&mut f, init_x);

        init_x = init_x_n;
        init_x -= grad.mul_scalar(lr);
    }

    init_x
}

pub mod layer;
pub mod model;
pub mod optimizer;
pub mod train;

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_numerical_gradient() {
        fn function_2(x: &Array<1, f32>) -> f32 {
            x[[0]].powi(2) + x[[1]].powi(2)
        }

        let x = Array::from_vec(vec![3.0, 4.0], [2]);

        let (grad, _x_n) = numerical_gradient(function_2, x);
        assert_relative_eq!(grad, Array::from_vec(vec![6.0, 8.0], [2]), epsilon = 2e-2);
    }
}
