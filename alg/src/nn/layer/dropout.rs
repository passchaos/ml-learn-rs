use burn_tensor::Bool;

use crate::nn::{Float, Tensor2, default_device, layer::LayerWard};

pub struct Dropout {
    ratio: Float,
    mask: Option<Tensor2<Bool>>,
}

impl Dropout {
    pub fn new(ratio: Float) -> Self {
        Dropout { ratio, mask: None }
    }
}

impl LayerWard for Dropout {
    fn forward(&mut self, input: Tensor2) -> Tensor2 {
        // if input.clone().contains_nan().into_scalar() == 1 {
        //     println!("dropout meet nan value");
        // }

        // 这里注意使用的是均匀分布，如果使用标准正态分布，那么会有很大比例的权重值被置为0，那就是捣乱了
        let mask1: burn_tensor::Tensor<_, _, burn_tensor::Float> = Tensor2::random(
            input.shape(),
            burn_tensor::Distribution::Default,
            &default_device(),
        );

        let mask = mask1.lower_elem(self.ratio);
        let v = input.mask_fill(mask.clone(), 0.0);

        self.mask = Some(mask);

        v
    }

    fn backward(&mut self, grad: Tensor2) -> Tensor2 {
        grad.mask_fill(self.mask.as_ref().unwrap().clone(), 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_forward() {
        let mut dropout = Dropout::new(0.2);

        let input = Tensor2::from_data(
            [
                [0.0, 0.2, 0.11, 0.13, 0.25],
                [-0.02, 0.03, 0.23, 0.58, 0.19],
            ],
            &default_device(),
        );

        let output = dropout.forward(input);
        println!("output: {output}");
    }
}
