use crate::nn::Mat;

pub mod batch_norm;
pub mod dropout;
pub mod linear;
pub mod relu;
pub mod sigmoid;
pub mod softmax_loss;

#[derive(Debug)]
pub enum Layer {
    Linear(linear::Linear),
    Relu(relu::Relu),
    Sigmoid(sigmoid::Sigmoid),
    Dropout(dropout::Dropout),
    BatchNorm(batch_norm::BatchNorm),
}

impl LayerWard for Layer {
    fn forward(&mut self, input: &Mat) -> Mat {
        match self {
            Layer::Linear(layer) => layer.forward(input),
            Layer::Relu(layer) => layer.forward(input),
            Layer::Sigmoid(layer) => layer.forward(input),
            Layer::Dropout(layer) => layer.forward(input),
            Layer::BatchNorm(layer) => layer.forward(input),
        }
    }
    fn backward(&mut self, grad: &Mat) -> Mat {
        match self {
            Layer::Linear(layer) => layer.backward(grad),
            Layer::Relu(layer) => layer.backward(grad),
            Layer::Sigmoid(layer) => layer.backward(grad),
            Layer::Dropout(layer) => layer.backward(grad),
            Layer::BatchNorm(layer) => layer.backward(grad),
        }
    }
}

pub trait LayerWard {
    fn forward(&mut self, input: &Mat) -> Mat;
    fn backward(&mut self, grad: &Mat) -> Mat;
}
