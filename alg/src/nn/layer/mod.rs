use std::fmt::Debug;

use num::Float;
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use vectra::{
    NumExt,
    prelude::{Array, Matmul},
};

pub mod batch_norm;
pub mod convolution;
pub mod dropout;
pub mod linear;
pub mod relu;
pub mod sigmoid;
pub mod softmax_loss;

#[derive(Debug)]
pub enum Layer<const D: usize, T: Debug + Float + NumExt> {
    Linear(linear::Linear<D, T>),
    Relu(relu::Relu<D, T>),
    Sigmoid(sigmoid::Sigmoid<D, T>),
    Dropout(dropout::Dropout<D, T>),
    BatchNorm(batch_norm::BatchNorm<D, T>),
}

pub trait LayerWard<const D1: usize, const D2: usize, T> {
    fn forward(&mut self, input: &Array<D1, T>) -> Array<D2, T>;
    fn backward(&mut self, grad: &Array<D2, T>) -> Array<D1, T>;
}

impl<T: Debug + Float + NumExt + SampleUniform> LayerWard<2, 2, T> for Layer<2, T>
where
    Array<2, T>: Matmul,
    StandardNormal: Distribution<T>,
{
    fn forward(&mut self, input: &Array<2, T>) -> Array<2, T> {
        match self {
            Layer::Linear(layer) => layer.forward(input),
            Layer::Relu(layer) => layer.forward(input),
            Layer::Sigmoid(layer) => layer.forward(input),
            Layer::Dropout(layer) => layer.forward(input),
            Layer::BatchNorm(layer) => layer.forward(input),
        }
    }
    fn backward(&mut self, grad: &Array<2, T>) -> Array<2, T> {
        match self {
            Layer::Linear(layer) => layer.backward(grad),
            Layer::Relu(layer) => layer.backward(grad),
            Layer::Sigmoid(layer) => layer.backward(grad),
            Layer::Dropout(layer) => layer.backward(grad),
            Layer::BatchNorm(layer) => layer.backward(grad),
        }
    }
}
