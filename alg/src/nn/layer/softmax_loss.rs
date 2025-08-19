use crate::{
    math::{SoftmaxOpT, loss::cross_entropy_error},
    nn::{Float, Mat},
};

#[derive(Default)]
pub struct SoftmaxWithLoss {
    y: Option<Mat>,
    t: Option<Mat>,
}

impl SoftmaxWithLoss {
    pub fn forward(&mut self, x: &Mat, t: &Mat) -> Float {
        self.t = Some(t.clone());

        let y = x.softmax();
        self.y = Some(y.clone());

        cross_entropy_error(y, t.clone())
    }

    pub fn backward(&mut self) -> Mat {
        let batch_size = self.t.as_ref().unwrap().shape()[0] as Float;

        let dx = (self.y.as_ref().unwrap() - self.t.as_ref().unwrap()) / batch_size;

        dx
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn test_softmax_with_loss_layer() {
        let y = array![[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]];
        let t = array![[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];

        let mut layer = SoftmaxWithLoss::default();
        let loss = layer.forward(&y, &t);
        let dx = layer.backward();

        assert_relative_eq!(loss, 1.8194936854234711, max_relative = 1e-7);
        assert_relative_eq!(
            dx,
            array![[
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
            ]],
            max_relative = 1e-7
        );
    }

    #[test]
    fn test_softmax_with_loss_layer_backward() {
        let y = array![
            [0.1, 0.05, 0.6, 0., 0.05, 0.1, 0., 0.1, 0., 0.],
            [0.1, 0.15, 0.5, 0., 0.05, 0.1, 0., 0.1, 0., 0.]
        ];
        let t = array![
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        ];

        let mut layer = SoftmaxWithLoss::default();
        let loss = layer.forward(&y, &t);
        let dx = layer.backward();

        assert_relative_eq!(loss, 2.066690565855486, max_relative = 1e-7);
        assert_relative_eq!(
            dx,
            array![
                [
                    0.04916165,
                    0.04676401,
                    -0.41894615,
                    0.0444833,
                    0.04676401,
                    0.04916165,
                    0.0444833,
                    0.04916165,
                    0.0444833,
                    0.0444833
                ],
                [
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
                ]
            ],
            max_relative = 1e-6
        );
    }
}
