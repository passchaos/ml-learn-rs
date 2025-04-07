use std::io::Cursor;

use anyhow::{Result, bail};
use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::Array2;
use safetensors::{View, tensor::TensorView};

pub trait Load {
    fn load(&self) -> Result<Array2<f32>>;
}

impl<'data> Load for TensorView<'data> {
    fn load(&self) -> Result<Array2<f32>> {
        let dtype = self.dtype();

        let mut data = Vec::new();
        match dtype {
            safetensors::Dtype::F32 => {
                let mut c_data = Cursor::new(self.data());

                let count = self.data_len() / dtype.size();

                for _ in 0..count {
                    let d = c_data.read_f32::<LittleEndian>()?;
                    data.push(d);
                }
            }
            _ => bail!("dtype is not supported: dtype= {dtype:?}"),
        };

        match self.shape() {
            [m, n] => {
                let f_d = Array2::from_shape_vec((*m, *n), data)?;
                Ok(f_d)
            }
            [m] => {
                let f_d = Array2::from_shape_vec((*m, 1), data)?;
                Ok(f_d)
            }
            d => {
                bail!("unsupported shape: {d:?}");
            }
        }
    }
}
