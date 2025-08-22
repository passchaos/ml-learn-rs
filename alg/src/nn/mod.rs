use burn_tensor::{DType, TensorData};
use safetensors::View;

type Float = f32;

#[cfg(feature = "alg-cuda")]
mod dt {
    use burn_cubecl::CubeBackend;

    pub type Device = cubecl::cuda::CudaDevice;
    type Runtime = cubecl::cuda::CudaRuntime;
    type BoolValue = u8;

    pub type Back<F = super::Float, I = i32> =
        burn_fusion::Fusion<CubeBackend<Runtime, F, I, BoolValue>>;
}

#[cfg(feature = "alg-wgpu")]
mod dt {
    use burn_cubecl::CubeBackend;

    pub type Device = cubecl::wgpu::WgpuDevice;
    type Runtime = cubecl::wgpu::WgpuRuntime;
    type BoolValue = u32;

    pub type Back<F = super::Float, I = i32> =
        burn_fusion::Fusion<CubeBackend<Runtime, F, I, BoolValue>>;
}

#[cfg(not(any(feature = "alg-cuda", feature = "alg-wgpu")))]
mod dt {
    use burn_ndarray::NdArray;

    pub type Device = burn_ndarray::NdArrayDevice;

    pub type Back<F = super::Float, I = i32> = NdArray<F, I>;
}

pub type Tensor1<K = burn_tensor::Float> = burn_tensor::Tensor<dt::Back, 1, K>;
pub type Tensor2<K = burn_tensor::Float> = burn_tensor::Tensor<dt::Back, 2, K>;

pub fn default_device() -> dt::Device {
    dt::Device::default()
}

pub fn float_epsilon() -> Float {
    1.0e-5
}

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

    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
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
