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

    // println!("orig y: {y}");
    let y = (y + crate::nn::float_epsilon()).log();

    // println!("y: {y} t: {t}");
    // let res = y.clone().contains_nan().into_scalar();
    // if res == 1 {
    //     panic!("meet nan data");
    // }

    let res = -(y * t).sum() / batch_size as Float;

    res.into_scalar()
}

impl SoftmaxWithLoss {
    pub fn forward(&mut self, x: Tensor2, t: Tensor2) -> Float {
        // if x.clone().contains_nan().into_scalar() == 1 {
        //     println!("softmax loss meet nan");
        // }

        self.t = Some(t.clone());

        // println!("before softmax: {x}");
        let y = burn_tensor::activation::softmax(x, 1);

        // println!("after softmax: {y}");
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
    use ndarray::{Array2, arr2, array};

    use crate::nn::default_device;

    use super::*;

    #[test]
    fn test_softmax_with_loss_layer() {
        let data1 = [[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]];
        let data2 = [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];

        let y1 = arr2(&data1);
        let t1 = arr2(&data2);

        let y = Tensor2::from_data(data1.clone(), &default_device());
        let ys = burn_tensor::activation::softmax(y.clone(), 1);

        let ys_data = [[
            0.09832329489873652,
            0.09352801122153914,
            0.1621077077048683,
            0.08896659628896098,
            0.09352801122153914,
            0.09832329489873652,
            0.08896659628896098,
            0.09832329489873652,
            0.08896659628896098,
            0.08896659628896098,
        ]];

        let ys_d: Tensor2 = Tensor2::from_data(ys_data, &default_device());
        assert!(ys.all_close(ys_d, None, None));

        let t = Tensor2::from_data(data2.clone(), &default_device());

        let mut layer = SoftmaxWithLoss::default();
        let loss = layer.forward(y, t);
        let dx = layer.backward();

        // assert_relative_eq!(loss, 1.8194936854234711, max_relative = 1e-7);

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
