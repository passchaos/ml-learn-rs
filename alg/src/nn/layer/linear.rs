use std::fmt::Debug;

use num::Float;
use num::cast;
use rand_distr::Distribution;
use rand_distr::StandardNormal;
use vectra::{
    NumExt,
    prelude::{Array, Matmul},
};

use crate::nn::{
    layer::LayerWard,
    optimizer::{Optimizer, OptimizerOpT},
};

#[derive(Clone, Copy, Debug)]
pub enum WeightInit<T> {
    Std(T),
    Xavier,
    He,
}

#[derive(Debug)]
pub struct Linear<const D: usize, T: Debug> {
    weight: Array<2, T>,
    bias: Option<Array<2, T>>,
    weight_opt: Optimizer<2, T>,
    bias_opt: Option<Optimizer<2, T>>,
    x: Option<Array<2, T>>,
    x_original_shape: Option<[usize; D]>,
}

impl<const D: usize, T: Debug + Float + NumExt> Linear<D, T>
where
    StandardNormal: Distribution<T>,
{
    pub fn new(
        weight_init: WeightInit<T>,
        input_size: usize,
        output_size: usize,
        weight_opt: Optimizer<2, T>,
        bias_opt: Option<Optimizer<2, T>>,
    ) -> Self {
        let weight = Array::randn([input_size, output_size]);

        let scale = match weight_init {
            WeightInit::Std(std) => std,
            WeightInit::Xavier => {
                let scale = (cast::<_, T>(6.0).unwrap()
                    / cast::<_, T>(input_size + output_size).unwrap())
                .sqrt();
                scale
            }
            WeightInit::He => {
                let scale = (cast::<_, T>(2.0).unwrap() / cast::<_, T>(input_size).unwrap()).sqrt();
                scale
            }
        };

        let weight = weight.mul_scalar(scale);

        let bias = if bias_opt.is_some() {
            Some(Array::zeros([1, output_size]))
        } else {
            None
        };

        Self {
            weight,
            bias,
            weight_opt,
            bias_opt,
            x: None,
            x_original_shape: None,
        }
    }
}

impl<const D: usize, T: Debug + Float + NumExt> LayerWard<D, 2, T> for Linear<D, T>
where
    StandardNormal: Distribution<T>,
    Array<2, T>: Matmul,
{
    fn forward(&mut self, input: Array<D, T>) -> Array<2, T> {
        self.x_original_shape = Some(input.shape());
        let new_shape = [input.shape()[0] as isize, -1];
        let input = input.clone().reshape(new_shape);

        self.x = Some(input.clone());

        let out = if let Some(bias) = &self.bias {
            &input.matmul(&self.weight) + bias
        } else {
            input.matmul(&self.weight)
        };

        out
    }

    fn backward(&mut self, grad: Array<2, T>) -> Array<D, T> {
        let dx = grad.matmul(&self.weight.clone().transpose());

        let x_t = self.x.as_ref().unwrap().clone().transpose();

        let dw = x_t.matmul(&grad);

        self.weight_opt.step(&mut self.weight, &dw);

        if let Some(bias) = self.bias.as_mut() {
            let db = grad.sum_axis(0);
            self.bias_opt.as_mut().unwrap().step(bias, &db);
        }

        dx.reshape(self.x_original_shape.unwrap().map(|i| i as isize))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use vectra::prelude::Array;

    use crate::nn::optimizer::Sgd;

    use super::*;

    #[test]
    fn test_linear_forward_backward() {
        let input_size = 4;
        let output_size = 3;
        let batch_size = 10;

        let weight_init = WeightInit::<f64>::He;
        let opt = Sgd::new(0.01);

        let weight = Array::from_vec(
            vec![
                -1.147584,
                0.53061247,
                -0.6196689,
                -0.64386505,
                -0.22075994,
                0.6821954,
                0.10052844,
                0.059097555,
                0.8720723,
                -0.6000754,
                -0.5773093,
                -0.56773186,
            ],
            [4, 3],
        );
        let bias = Array::from_vec(vec![0.0, 0.0, 0.0], [1, 3]);

        let mut linear = Linear {
            weight,
            bias: Some(bias),
            x: None,
            weight_opt: Optimizer::Sgd(opt.clone()),
            bias_opt: Some(Optimizer::Sgd(opt)),
            x_original_shape: None,
        };
        println!("linear: {linear:?}");

        let input = Array::from_vec(
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

        let output = linear.forward(input);
        let grad = linear.backward(output.clone());

        let output_r = Array::from_vec(
            vec![
                1.5798755,
                -0.25766864,
                1.9120206,
                -0.49872115,
                1.3232852,
                -0.9620689,
                0.77520555,
                0.7067366,
                -0.73724943,
                1.0267,
                -1.153101,
                2.6188345,
                0.39989066,
                -0.6985538,
                -1.1720206,
                0.841953,
                0.63980377,
                0.9806836,
                1.1075634,
                -0.8091126,
                -0.80791646,
                3.024921,
                -0.69801944,
                0.681222,
                1.6192747,
                -0.9120107,
                -1.7385087,
                -2.778898,
                -0.1143061,
                -1.5029051,
            ],
            [10, 3],
        );
        let grad_r = Array::from_vec(
            vec![
                -3.1345816,
                0.34402803,
                1.811015,
                -1.884805,
                1.8706403,
                -0.6273383,
                -0.8109264,
                0.08152261,
                -0.057759624,
                -1.1580951,
                -0.5232382,
                -0.45462742,
                -3.4128845,
                1.3800591,
                2.31888,
                -1.4371973,
                -0.10330477,
                -0.90281,
                -1.023169,
                0.82871044,
                -1.234423,
                -0.0143292835,
                0.97767806,
                -1.4313654,
                -1.1997064,
                -1.0856586,
                -0.6410365,
                0.26116654,
                -4.2638607,
                -1.3288196,
                0.85691416,
                -1.7989589,
                -1.264878,
                -2.0272617,
                -1.4072196,
                0.5418321,
                4.05967,
                0.7891946,
                -1.5967554,
                2.5867856,
            ],
            [10, 4],
        );
        println!("output= {output:?} grad= {grad:?}");

        assert_relative_eq!(output, output_r);
        assert_relative_eq!(grad, grad_r);
    }
}
