use std::fmt::Debug;

use num::Float;
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use vectra::{
    NumExt,
    prelude::{Array, Matmul},
};

use crate::nn::{
    layer::{
        Layer, LayerWard,
        batch_norm::BatchNorm,
        dropout::Dropout,
        linear::{Linear, WeightInit},
        relu::Relu,
        softmax_loss::SoftmaxWithLoss,
    },
    optimizer::Optimizer,
};

// mod layer;
// mod simple;
// mod two_layer;

#[derive(Debug)]
pub struct Model<T: Float + NumExt> {
    layers: Vec<Layer<2, T>>,
    out: SoftmaxWithLoss<T>,
}

impl<T: Debug + Float + NumExt + SampleUniform> Model<T>
where
    StandardNormal: Distribution<T>,
    Array<2, T>: Matmul,
{
    pub fn new(
        input_size: usize,
        hidden_sizes: &[usize],
        output_size: usize,
        weight_init: WeightInit<T>,
        batch_norm_momentum: Option<T>,
        dropout_ratio: Option<T>,
        opt: Optimizer<2, T>,
    ) -> Self {
        let mut layers = vec![];

        let mut inner_input_size = input_size;

        for size in hidden_sizes {
            let inner_output_size = *size;

            let lin = Linear::new(
                weight_init,
                inner_input_size,
                inner_output_size,
                opt.clone(),
                Some(opt.clone()),
            );
            layers.push(Layer::Linear(lin));

            if let Some(ratio) = dropout_ratio {
                let dropout = Dropout::new(ratio);
                layers.push(Layer::Dropout(dropout));
            };

            if let Some(momentum) = batch_norm_momentum {
                let gamma = Array::ones([1, inner_output_size]);
                let beta = Array::zeros([1, inner_output_size]);
                let batch_norm = BatchNorm::new(gamma, beta, momentum, opt.clone());

                layers.push(Layer::BatchNorm(batch_norm));
            }
            let relu = Relu::default();
            layers.push(Layer::Relu(relu));

            inner_input_size = inner_output_size;
        }

        let lin2 = Linear::new(
            weight_init,
            inner_input_size,
            output_size,
            opt.clone(),
            Some(opt),
        );
        layers.push(Layer::Linear(lin2));

        let out = SoftmaxWithLoss::default();

        Self { layers, out }
    }

    pub fn predict(&mut self, x: &Array<2, T>) -> Array<2, T> {
        let mut x = x.clone();

        for layer in &mut self.layers {
            x = layer.forward(&x);
        }

        x
    }

    pub fn loss(&mut self, x: &Array<2, T>, t: &Array<2, T>) -> T {
        let y = self.predict(x);

        self.out.forward(&y, t)
    }

    pub fn backward(&mut self) {
        let mut dout = self.out.backward();

        for layer in self.layers.iter_mut().rev() {
            dout = layer.backward(&dout);
        }
    }
}
