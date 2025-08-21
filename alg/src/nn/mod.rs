use burn_cubecl::CubeBackend;
use burn_tensor::backend::Backend;
use cubecl::cuda::{CudaDevice, CudaRuntime};

type Float = f32;

type Cuda<F = Float, I = i32> = burn_fusion::Fusion<CubeBackend<CudaRuntime, F, I, u8>>;

pub type Tensor2<K = burn_tensor::Float> = burn_tensor::Tensor<Cuda, 2, K>;
pub type Tensor1<K = burn_tensor::Float> = burn_tensor::Tensor<Cuda, 1, K>;
pub fn float_epsilon() -> Float {
    1.0e-5
}

pub fn default_device() -> CudaDevice {
    CudaDevice::default()
}

pub mod dataset;
pub mod layer;
pub mod model;
pub mod optimizer;
pub mod train;
