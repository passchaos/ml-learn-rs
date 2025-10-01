use ndarray::{Array, Dimension, NdFloat};

pub fn mean_squared_error<D: Dimension>(y: &Array<f32, D>, t: &Array<f32, D>) -> f32 {
    (y - t).mapv(|x| x.powi(2)).sum() / 2.0
}

pub fn cross_entropy_error<T: NdFloat + From<f32>, D: Dimension>(
    y: Array<T, D>,
    t: Array<T, D>,
) -> T {
    let mut y = y.into_dyn();
    let mut t = t.into_dyn();

    let delta = 1e-7;

    if y.ndim() == 1 {
        let len = t.len();
        t = t.into_shape_with_order((1, len)).unwrap().into_dyn();
        y = y.into_shape_with_order((1, len)).unwrap().into_dyn();
    }

    let batch_size = y.shape()[0] as f32;

    let y1 = y.mapv(|a| a + delta.into());
    let y1 = y1.ln();

    let mut v1 = t * y1;
    v1.mapv_inplace(|a| a * (-1.0).into());
    v1.sum() / batch_size.into()
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn test_cross_entropy_error() {
        let t = array![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let y = array![0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0];

        let err1 = mean_squared_error(&y, &t);
        let err2 = cross_entropy_error(y, t);

        println!("mse= {err1} cee= {err2}");
        assert_relative_eq!(err1, 0.097500000000000031);
        assert_relative_eq!(err2, 0.510825457, epsilon = 1e-6);
    }
}
