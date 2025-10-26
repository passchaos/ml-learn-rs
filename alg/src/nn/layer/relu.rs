use num::Float;
use vectra::{NumExt, prelude::Array};

use crate::nn::layer::LayerWard;

#[derive(Default, Debug)]
pub struct Relu<const D: usize, T: Float + NumExt> {
    mask: Option<Array<D, bool>>,
    phantom: std::marker::PhantomData<T>,
}

impl<const D: usize, T: Float + NumExt> LayerWard<D, D, T> for Relu<D, T> {
    fn forward(&mut self, x: Array<D, T>) -> Array<D, T> {
        self.mask = Some(x.map(|&a| a <= T::zero()));

        let res = x.relu();

        res
    }

    fn backward(&mut self, mut grad: Array<D, T>) -> Array<D, T> {
        grad.multi_iter_mut(|idx, item| {
            let v = self.mask.as_ref().unwrap()[idx.map(|a| a as isize)];

            if v {
                *item = T::zero();
            }
        });

        grad
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use vectra::prelude::Array;

    use super::*;

    #[test]
    fn test_relu_layer() {
        let mut relu = Relu::default();
        let x = Array::from_vec(
            vec![
                -1.1075944,
                0.16940418,
                0.91992205,
                -0.54228693,
                1.0911559,
                -0.06274483,
                -1.1812001,
                -1.3861872,
                0.27621895,
                -1.2316114,
                -0.011528776,
                -0.5005328,
                -1.4678969,
                0.80560935,
                1.6619517,
                0.5102769,
                -0.55237263,
                -0.5267347,
                -0.788604,
                0.82301694,
                -0.09303126,
                -0.015436283,
                0.31842378,
                -1.1552585,
                -0.9052046,
                -1.0068942,
                -0.17200667,
                0.93696123,
                -2.0976977,
                -0.51475143,
                -0.6927939,
                -0.5930114,
                -1.2155122,
                -1.3309227,
                -1.2681531,
                0.8416884,
                1.4421065,
                0.48525217,
                -0.22205828,
                1.3151674,
            ],
            [10, 4],
        );

        let output = relu.forward(x.clone());
        let grad = relu.backward(output.clone());
        println!("x= {x:?} output= {output:?} grad= {grad:?}");

        let output_r = Array::from_vec(
            vec![
                0.0, 0.16940418, 0.91992205, 0.0, 1.0911559, 0.0, 0.0, 0.0, 0.27621895, 0.0, 0.0,
                0.0, 0.0, 0.80560935, 1.6619517, 0.5102769, 0.0, 0.0, 0.0, 0.82301694, 0.0, 0.0,
                0.31842378, 0.0, 0.0, 0.0, 0.0, 0.93696123, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.8416884, 1.4421065, 0.48525217, 0.0, 1.3151674,
            ],
            [10, 4],
        );

        assert_relative_eq!(output, output_r);
    }
}
