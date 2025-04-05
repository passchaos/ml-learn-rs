use ndarray::{Array1, Array2};

pub mod safetensors;

#[derive(Debug)]
pub enum Tensor {
    Dim1F32(Array1<f32>),
    Dim2F32(Array2<f32>),
}
