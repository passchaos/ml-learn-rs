use std::fmt::Debug;

use rand_distr::uniform::SampleUniform;
use vectra::{NumExt, prelude::Array};

use crate::nn::layer::LayerWard;

#[derive(Debug)]
pub struct Dropout<const D: usize, T: Debug> {
    ratio: T,
    mask: Option<Array<D, T>>,
}

impl<const D: usize, T: Debug> Dropout<D, T> {
    pub fn new(ratio: T) -> Self {
        Dropout { ratio, mask: None }
    }
}

impl<const D: usize, T: Debug + NumExt + SampleUniform + PartialOrd> LayerWard<D, D, T>
    for Dropout<D, T>
{
    fn forward(&mut self, input: &Array<D, T>) -> Array<D, T> {
        // 这里注意使用的是均匀分布，如果使用标准正态分布，那么会有很大比例的权重值被置为0，那就是捣乱了
        let mask = Array::<D, T>::random(input.shape())
            .map_into(|x| if x < self.ratio { T::zero() } else { T::one() });

        let v = &mask * input;
        self.mask = Some(mask);

        v
    }

    fn backward(&mut self, grad: &Array<D, T>) -> Array<D, T> {
        self.mask.as_ref().unwrap() * grad
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_forward() {
        let rand_v = Array::<_, f64>::randn([2, 5]);
        println!("rand v: {rand_v}");

        let mut dropout = Dropout::new(0.2);
        let input = Array::from_vec(
            vec![0.0, 0.2, 0.11, 0.13, 0.25, -0.02, 0.03, 0.23, 0.58, 0.19],
            [2, 5],
        );

        let output = dropout.forward(&input);
        println!("output: {output}");
    }
}
