use ndarray::{ArrayBase, RawData};

use crate::nn::{Mat, optimizer::Optimizer};

pub mod linear;
pub mod relu;
pub mod sigmoid;
pub mod softmax_loss;

pub trait Layer {
    fn forward(&mut self, input: &Mat) -> Mat;
    fn backward<O: Optimizer>(&mut self, grad: &Mat, opt: &mut O) -> Mat;
}
