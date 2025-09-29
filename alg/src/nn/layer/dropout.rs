use crate::nn::{Float, Mat, layer::LayerWard};

pub struct Dropout {
    ratio: Float,
    mask: Option<Mat>,
}

impl Dropout {
    pub fn new(ratio: Float) -> Self {
        Dropout { ratio, mask: None }
    }
}

impl LayerWard for Dropout {
    fn forward(&mut self, input: &Mat) -> Mat {
        // 这里注意使用的是均匀分布，如果使用标准正态分布，那么会有很大比例的权重值被置为0，那就是捣乱了
        let mask =
            rand::<_, _, Float>(input.raw_dim()).map(|x| if x < &self.ratio { 0.0 } else { 1.0 });

        let v = &mask * input;
        self.mask = Some(mask);

        v
    }

    fn backward(&mut self, grad: &Mat) -> Mat {
        self.mask.as_ref().unwrap() * grad
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    use super::*;

    #[test]
    fn test_dropout_forward() {
        let rand_v = rand::<_, _, f32>((2, 5));
        println!("rand v: {rand_v}");

        let mut dropout = Dropout::new(0.2);
        let input = arr2(&[
            [0.0, 0.2, 0.11, 0.13, 0.25],
            [-0.02, 0.03, 0.23, 0.58, 0.19],
        ]);

        let output = dropout.forward(&input);
        println!("output: {output}");
    }
}
