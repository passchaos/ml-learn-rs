use crate::{
    math::SoftmaxOpT,
    nn::{Float, Tensor2},
};

#[derive(Default)]
pub struct SoftmaxWithLoss {
    y: Option<Tensor2>,
    t: Option<Tensor2>,
}

fn cross_entropy_error(y: Tensor2, t: Tensor2) -> Float {
    let batch_size = y.dims()[0];

    let y = (y + crate::nn::float_epsilon()).log();

    println!("y: {y} t: {t}");
    let res = -(y * t).sum() / batch_size as Float;

    res.into_scalar()
}

impl SoftmaxWithLoss {
    pub fn forward(&mut self, x: Tensor2, t: Tensor2) -> Float {
        self.t = Some(t.clone());

        let y = burn_tensor::activation::softmax(x, 1);
        self.y = Some(y.clone());

        cross_entropy_error(y, t.clone())
    }

    pub fn backward(&mut self) -> Tensor2 {
        let batch_size = self.t.as_ref().unwrap().dims()[0] as Float;

        let dx = (self.y.as_ref().unwrap().clone() - self.t.as_ref().unwrap().clone()) / batch_size;

        dx
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::array;

    use crate::nn::default_device;

    use super::*;

    #[test]
    fn test_softmax_with_loss_layer() {
        let y = Tensor2::from_data(
            [[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]],
            &default_device(),
        );
        let t = Tensor2::from_data(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            &default_device(),
        );

        let mut layer = SoftmaxWithLoss::default();
        let loss = layer.forward(y, t);
        let dx = layer.backward();

        assert_relative_eq!(loss, 1.8194936854234711, max_relative = 1e-7);

        let check_v = Tensor2::from_data(
            [[
                0.09832329,
                0.09352801,
                -0.83789229,
                0.0889666,
                0.09352801,
                0.09832329,
                0.0889666,
                0.09832329,
                0.0889666,
                0.0889666,
            ]],
            &default_device(),
        );

        assert!(dx.all_close(check_v, None, None));
    }

    #[test]
    fn test_softmax_with_loss_layer_backward() {
        let y = Tensor2::from_data(
            [
                [0.1, 0.05, 0.6, 0., 0.05, 0.1, 0., 0.1, 0., 0.],
                [0.1, 0.15, 0.5, 0., 0.05, 0.1, 0., 0.1, 0., 0.],
            ],
            &default_device(),
        );
        let t = Tensor2::from_data(
            [
                [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            ],
            &default_device(),
        );

        let mut layer = SoftmaxWithLoss::default();
        let loss = layer.forward(y, t);
        let dx = layer.backward();

        assert_relative_eq!(loss, 2.066690565855486, max_relative = 1e-7);
        let v = Tensor2::from_data(
            [
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
                    0.0444833,
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
                    0.04473336,
                ],
            ],
            &default_device(),
        );

        assert!(dx.all_close(v, None, None));
    }
}
