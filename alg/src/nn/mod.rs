use burn_cubecl::CubeBackend;
use burn_tensor::{DType, TensorData, backend::Backend};
use cubecl::cuda::{CudaDevice, CudaRuntime};
use safetensors::View;

type Float = f32;

type Cuda<F = Float, I = i32> = burn_fusion::Fusion<CubeBackend<CudaRuntime, F, I, u8>>;

pub type Tensor2<K = burn_tensor::Float> = burn_tensor::Tensor<Cuda, 2, K>;

#[derive(Debug)]
pub struct Tensor2Data(TensorData);

impl Tensor2Data {
    pub fn to_tensor(self) -> Tensor2 {
        Tensor2::from_data(self.0, &default_device())
    }

    pub fn from_tensor(tensor: Tensor2) -> Self {
        Tensor2Data(tensor.into_data())
    }

    pub fn extract_tensor_from_st(st: &safetensors::SafeTensors, name: &str) -> Tensor2 {
        let tensor = st.tensor(name).unwrap();
        Tensor2Data(TensorData::from_bytes(
            tensor.data().to_vec(),
            tensor.shape(),
            DType::F32,
        ))
        .to_tensor()
    }
}

pub type Tensor1<K = burn_tensor::Float> = burn_tensor::Tensor<Cuda, 1, K>;
pub fn float_epsilon() -> Float {
    1.0e-5
}

pub fn default_device() -> CudaDevice {
    CudaDevice::default()
}

impl View for Tensor2Data {
    fn dtype(&self) -> safetensors::Dtype {
        match self.0.dtype {
            burn_tensor::DType::F32 => safetensors::Dtype::F32,
            _ => panic!("Unsupported dtype"),
        }
    }

    fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    fn data(&self) -> std::borrow::Cow<[u8]> {
        self.0.as_bytes().into()
    }

    fn data_len(&self) -> usize {
        self.data().len()
    }
}

pub mod dataset;
pub mod layer;
pub mod model;
pub mod optimizer;
pub mod train;
