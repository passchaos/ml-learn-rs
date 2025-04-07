use ndarray::{Array, Dimension};

pub fn mean_squared_error<D: Dimension>(y: &Array<f32, D>, t: &Array<f32, D>) -> f32 {
    (y - t).mapv(|x| x.powi(2)).sum() / 2.0
}

pub fn cross_entropy_error<D: Dimension>(y: &Array<f32, D>, t: &Array<f32, D>) -> f32 {
    let delta = 1e-7;
    -1.0 * ((t * (y + delta).ln()).sum())
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
        let err2 = cross_entropy_error(&y, &t);

        println!("mse= {err1} cee= {err2}");
        assert_relative_eq!(err1, 0.097500000000000031);
        assert_relative_eq!(err2, 0.510825457, epsilon = 1e-6);
    }
}
