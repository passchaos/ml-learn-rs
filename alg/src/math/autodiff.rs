use std::ops::{Add, Div, Mul, Sub, SubAssign};

use ndarray::{Array, Dimension, NdIndex};

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
    T: Sub<Output = T> + Add<Output = T> + Div<Output = T> + Copy,
    D: Dimension,
>(
    f: impl Fn(Array<T, D>) -> T,
    x: Array<T, D>,
    delta: T,
) -> Array<T, D>
where
    D::Pattern: NdIndex<D>,
{
    let mut res = x.clone();

    for (idx, v) in x.clone().indexed_iter() {
        let mut new_x_1 = x.clone();
        new_x_1[idx.clone()] = *v + delta;

        let mut new_x_2 = x.clone();
        new_x_2[idx.clone()] = *v - delta;

        res[idx] = (f(new_x_1) - f(new_x_2)) / (delta + delta);
    }

    res
}

pub fn gradient_descent<
    T: Sub<Output = T> + Add<Output = T> + Div<Output = T> + Mul<Output = T> + SubAssign + Copy,
    D: Dimension,
>(
    f: fn(Array<T, D>) -> T,
    delta: T,
    init_x: Array<T, D>,
    lr: T,
    step_num: usize,
) -> Array<T, D>
where
    D::Pattern: NdIndex<D>,
{
    let mut x = init_x;

    for _ in 0..step_num {
        let mut grad = numerical_gradient(f, x.clone(), delta);
        grad.mapv_inplace(|a| a * lr);

        x -= &grad;
    }

    x
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
        let f1 = |x: Array1<f64>| x.pow2().sum();

        let g1 = numerical_gradient(f1, array![3.0, 4.0], 1e-4);
        let g2 = numerical_gradient(f1, array![0.0, 2.0], 1e-4);
        let g3 = numerical_gradient(f1, array![3.0, 0.0], 1e-4);
        println!("g1: {g1} g2= {g2} g3= {g3}");

        let f2 = |x: Array2<f64>| x.pow2().sum();
        let g4 = numerical_gradient(f2, array![[1.0, 2.0], [3.0, 4.0]], 1e-4);
        println!("g4: {g4}");
    }

    #[test]
    fn test_gradient_descent() {
        let f = |x: Array1<f32>| x.pow2().sum();

        let x = gradient_descent(f, 1e-4, array![-3.0, 4.0], 0.1, 100);
        println!("x: {x}");

        assert_relative_eq!(x, array![0.0, 0.0]);
    }
}
