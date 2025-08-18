use crate::nn::{Mat, optimizer::Optimizer};

pub mod batch_norm;
pub mod linear;
pub mod relu;
pub mod sigmoid;
pub mod softmax_loss;

pub enum Layer {
    Linear(linear::Linear),
    Relu(relu::Relu),
    Sigmoid(sigmoid::Sigmoid),
}

impl LayerWard for Layer {
    fn forward(&mut self, input: &Mat) -> Mat {
        match self {
            Layer::Linear(layer) => layer.forward(input),
            Layer::Relu(layer) => layer.forward(input),
            Layer::Sigmoid(layer) => layer.forward(input),
        }
    }
    fn backward(&mut self, grad: &Mat) -> Mat {
        match self {
            Layer::Linear(layer) => layer.backward(grad),
            Layer::Relu(layer) => layer.backward(grad),
            Layer::Sigmoid(layer) => layer.backward(grad),
        }
    }
}

pub trait LayerWard {
    fn forward(&mut self, input: &Mat) -> Mat;
    fn backward(&mut self, grad: &Mat) -> Mat;
}
