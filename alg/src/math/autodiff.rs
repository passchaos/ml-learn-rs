use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Sub, SubAssign},
};

use ndarray::{Array, Dimension, NdIndex};
use rand::TryRngCore;

pub fn numerical_diff<T: Sub<Output = T> + Add<Output = T> + Div<Output = T> + Copy>(
    f: fn(T) -> T,
    x: T,
    delta: T,
) -> T {
    let a = f(x + delta) - f(x - delta);
    let b = delta + delta;

    a / b
}

pub fn numerical_gradient<
    T: Sub<Output = T> + Add<Output = T> + Div<Output = T> + Copy + Debug,
    D: Dimension,
    F: FnMut(&Array<T, D>) -> T,
>(
    mut f: F,
    x: &mut Array<T, D>,
    delta: T,
) -> Array<T, D>
where
    D::Pattern: NdIndex<D>,
{
    let mut res = x.clone();

    let idx_iter: Vec<_> = x.indexed_iter().map(|(idx, _)| idx).clone().collect();

    for idx in idx_iter {
        let tmp_val = x[idx.clone()];

        x[idx.clone()] = tmp_val + delta;
        let fxh1 = f(x);

        x[idx.clone()] = tmp_val - delta;
        let fxh2 = f(x);

        res[idx.clone()] = (fxh1 - fxh2) / (delta + delta);

        x[idx] = tmp_val;
    }

    res
}

pub fn gradient_descent<
    T: Sub<Output = T>
        + Add<Output = T>
        + Div<Output = T>
        + Mul<Output = T>
        + SubAssign
        + Copy
        + Debug,
    D: Dimension,
>(
    f: fn(&Array<T, D>) -> T,
    delta: T,
    x: &mut Array<T, D>,
    lr: T,
    step_num: usize,
) where
    D::Pattern: NdIndex<D>,
{
    for _ in 0..step_num {
        let mut grad = numerical_gradient(f, x, delta);
        println!("grad: {grad:?}");

        grad.mapv_inplace(|a| a * lr);

        *x = &*x - &grad;
    }
}

#[cfg(test)]
mod tests {
    use std::f32;

    use approx::assert_relative_eq;
    use ndarray::{Array1, Array2, array};

    use super::*;

    #[test]
    fn test_numerical_diff() {
        let f = |x: f64| x.powi(2);

        let a = numerical_diff(f, 10.0, 1e-4);
        println!("a: {a}");

        assert_relative_eq!(a, 20.0, max_relative = 0.0001);
    }

    #[test]
    fn test_numerical_gradient() {
        let f1 = |x: &Array1<f64>| x.pow2().sum();

        let g1 = numerical_gradient(f1, &mut array![3.0, 4.0], 1e-4);
        let g2 = numerical_gradient(f1, &mut array![0.0, 2.0], 1e-4);
        let g3 = numerical_gradient(f1, &mut array![3.0, 0.0], 1e-4);
        assert_relative_eq!(g1, array![6.0, 8.0], max_relative = 0.0001);
        assert_relative_eq!(g2, array![0.0, 4.0], max_relative = 0.0001);
        assert_relative_eq!(g3, array![6.0, 0.0], max_relative = 0.0001);
        println!("g1: {g1} g2= {g2} g3= {g3}");

        let f2 = |x: &Array2<f64>| x.pow2().sum();
        let g4 = numerical_gradient(f2, &mut array![[1.0, 2.0], [3.0, 4.0]], 1e-4);

        assert_relative_eq!(g4, array![[2.0, 4.0], [6.0, 8.0]], max_relative = 0.0001);
        println!("g4: {g4}");
    }

    #[test]
    fn test_gradient_descent() {
        let f = |x: &Array1<f32>| x.pow2().sum();

        let mut x = array![-3.0, 4.0];
        gradient_descent(f, 1e-4, &mut x, 0.1, 100);
        println!("x: {x}");

        assert_relative_eq!(x, array![0.0, 0.0]);
    }
}
