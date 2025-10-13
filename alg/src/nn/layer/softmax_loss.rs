use num::{Float, cast};
use vectra::{NumExt, prelude::Array};

use crate::math::{ActivationFn, LossFn};

#[derive(Default, Debug)]
pub struct SoftmaxWithLoss<T: Float + NumExt> {
    y: Option<Array<2, T>>,
    t: Option<Array<2, T>>,
}

impl<T: Float + NumExt> SoftmaxWithLoss<T> {
    pub fn forward(&mut self, x: &Array<2, T>, t: &Array<2, T>) -> T {
        self.t = Some(t.clone());

        let y = x.softmax();
        self.y = Some(y.clone());

        y.cross_entropy_error(t)
    }

    pub fn backward(&mut self) -> Array<2, T> {
        let batch_size = self.t.as_ref().unwrap().shape()[0];
        let batch_size = cast(batch_size).unwrap();

        (self.y.as_ref().unwrap() - self.t.as_ref().unwrap()).div_scalar(batch_size)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_softmax_with_loss_layer() {
        let y = Array::<2, f32>::from_vec(
            vec![0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],
            [1, 10],
        );
        let t = Array::<2, f32>::from_vec(
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1, 10],
        );

        let mut layer = SoftmaxWithLoss::default();
        let loss = layer.forward(&y, &t);
        let dx = layer.backward();

        assert_relative_eq!(loss, 1.8194936854234711, epsilon = 1e-5);
        assert_relative_eq!(
            dx,
            Array::<2, f32>::from_vec(
                vec![
                    0.09832329,
                    0.09352801,
                    -0.83789229,
                    0.0889666,
                    0.09352801,
                    0.09832329,
                    0.0889666,
                    0.09832329,
                    0.0889666,
                    0.0889666
                ],
                [1, 10]
            ),
            max_relative = 1e-7
        );
    }

    #[test]
    fn test_softmax_with_loss_layer_backward() {
        let y = Array::<2, f32>::from_vec(
            vec![
                0.1, 0.05, 0.6, 0., 0.05, 0.1, 0., 0.1, 0., 0., 0.1, 0.15, 0.5, 0., 0.05, 0.1, 0.,
                0.1, 0., 0.,
            ],
            [2, 10],
        );
        let t = Array::<2, f32>::from_vec(
            vec![
                0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [2, 10],
        );

        let mut layer = SoftmaxWithLoss::default();
        let loss = layer.forward(&y, &t);
        let dx = layer.backward();

        assert_relative_eq!(loss, 2.066690565855486, epsilon = 1e-5);
        assert_relative_eq!(
            dx,
            Array::<2, f32>::from_vec(
                vec![
                    0.04916165,
                    0.04676401,
                    -0.41894615,
                    0.0444833,
                    0.04676401,
                    0.04916165,
                    0.0444833,
                    0.04916165,
                    0.0444833,
                    0.0444833,
                    -0.45056199,
                    0.05197276,
                    0.07375285,
                    0.04473336,
                    0.04702689,
                    0.04943801,
                    0.04473336,
                    0.04943801,
                    0.04473336,
                    0.04473336
                ],
                [2, 10]
            ),
            max_relative = 1e-6
        );
    }
}
