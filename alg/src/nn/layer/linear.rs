use std::collections::HashMap;

use ndarray::Axis;

use crate::{
    math::stat::randn,
    nn::{
        Float, Tensor2, Tensor2Data, default_device,
        layer::LayerWard,
        optimizer::{Optimizer, OptimizerOpT},
    },
};

#[derive(Clone, Copy, Debug)]
pub enum WeightInit {
    Std(Float),
    Xavier,
    He,
}

pub struct Linear {
    weight: Tensor2,
    bias: Option<Tensor2>,
    weight_opt: Optimizer,
    bias_opt: Option<Optimizer>,
    x: Option<Tensor2>,
}

impl Linear {
    pub fn new(
        weight_init: WeightInit,
        input_size: usize,
        output_size: usize,
        weight_opt: Optimizer,
        bias_opt: Option<Optimizer>,
    ) -> Self {
        let weight = Tensor2::random(
            [input_size, output_size],
            burn_tensor::Distribution::Normal(0.0, 1.0),
            &default_device(),
        );

        let scale = match weight_init {
            WeightInit::Std(std) => std,
            WeightInit::Xavier => {
                let scale = (6.0 / (input_size + output_size) as Float).sqrt();
                scale
            }
            WeightInit::He => {
                let scale = (2.0 / input_size as Float).sqrt();
                scale
            }
        };

        let weight = weight * scale;

        let bias = if bias_opt.is_some() {
            Some(Tensor2::zeros([1, output_size], &default_device()))
        } else {
            None
        };

        Self {
            weight,
            bias,
            weight_opt,
            bias_opt,
            x: None,
        }
    }
}

impl LayerWard for Linear {
    fn forward(&mut self, input: Tensor2) -> Tensor2 {
        // if input.clone().contains_nan().into_scalar() == 1 {
        //     println!("linear meet nan value");
        // }

        self.x = Some(input.clone());

        let res = if let Some(bias) = &self.bias {
            input.matmul(self.weight.clone()) + bias.clone()
        } else {
            input.matmul(self.weight.clone())
        };

        // if res.clone().contains_nan().into_scalar() == 1 {
        //     let new_res = self.x.as_ref().unwrap().clone().matmul(self.weight.clone())
        //         + self.bias.as_ref().unwrap().clone();
        //     let mut f = std::fs::File::create("abc.bin").unwrap();

        //     let input_sum = self.x.as_ref().unwrap().clone().sum().into_scalar();

        //     let mut contents = HashMap::new();
        //     contents.insert(
        //         "input",
        //         Tensor2Data::from_tensor(self.x.as_ref().unwrap().clone()),
        //     );
        //     contents.insert("weight", Tensor2Data::from_tensor(self.weight.clone()));
        //     contents.insert(
        //         "bias",
        //         Tensor2Data::from_tensor(self.bias.as_ref().unwrap().clone()),
        //     );

        //     println!("input_sum= {input_sum} res= {res} new_res= {new_res}");
        //     println!("contents: {contents:?}");
        //     let path = std::path::Path::new("abc.safetensors");

        //     std::fs::remove_file(path);
        //     safetensors::serialize_to_file(contents, None, path).unwrap();

        //     println!("linear output nan value");
        //     println!(
        //         "input: {} weight= {} bias= {}",
        //         self.x.as_ref().unwrap(),
        //         self.weight,
        //         self.bias.as_ref().unwrap()
        //     );

        //     panic!("meet nan first");
        // }

        res
    }

    fn backward(&mut self, grad: Tensor2) -> Tensor2 {
        let dx = grad.clone().matmul(self.weight.clone().transpose());

        let dw = self
            .x
            .as_ref()
            .unwrap()
            .clone()
            .transpose()
            .matmul(grad.clone());

        self.weight_opt.step(&mut self.weight, dw);

        if let Some(bias) = self.bias.as_mut() {
            let db = grad.sum_dim(0);
            self.bias_opt.as_mut().unwrap().step(bias, db);
        }

        dx
    }
}

#[cfg(test)]
mod tests {
    use burn_tensor::TensorData;

    use crate::nn::{Tensor2, Tensor2Data, default_device};

    #[test]
    fn test_matmul_nan() {
        let file = std::fs::read("/tmp/abc1.safetensors").unwrap();

        let info = safetensors::SafeTensors::read_metadata(&file).unwrap();
        println!("info: {info:?}");
        let res = safetensors::SafeTensors::deserialize(&file).unwrap();

        let input = Tensor2Data::extract_tensor_from_st(&res, "input");
        let weight = Tensor2Data::extract_tensor_from_st(&res, "weight");
        let bias = Tensor2Data::extract_tensor_from_st(&res, "bias");

        println!("input: {input}");
        println!("weight: {weight}");
        println!("bias: {bias}");

        let res = input.matmul(weight) + bias;
        println!("res: {res}");
        return;

        let input = Tensor2::zeros([10000, 784], &default_device());

        for _ in 0..10000 {
            let weight = Tensor2::random(
                [784, 100],
                burn_tensor::Distribution::Normal(0.0, 1.0),
                &default_device(),
            );

            let bias = Tensor2::random(
                [1, 100],
                burn_tensor::Distribution::Normal(0.0, 1.0),
                &default_device(),
            );

            let res = input.clone().matmul(weight) + bias;

            if res.contains_nan().into_scalar() == 1 {
                panic!("NaN detected");
            }
        }
    }
}
