use std::ops::Mul;
use std::ops::Sub;

use ndarray::Array;
use ndarray::ArrayBase;
use ndarray::Data;
use ndarray::DimMax;
use ndarray::Dimension;
use ndarray::RawData;
use ndarray::ScalarOperand;

pub trait Optimizer<S: RawData, D, A> {
    fn update(&mut self, params: &mut [(&mut Array<A, D>, &ArrayBase<S, D>)]);
}

#[derive(Debug)]
pub struct Sgd<T> {
    lr: T,
}

impl<
    T: ScalarOperand,
    A: Clone + Mul<T, Output = A> + Sub<A, Output = A>,
    S: Data<Elem = A>,
    D: Dimension + DimMax<D>,
> Optimizer<S, D, A> for Sgd<T>
{
    fn update(&mut self, params: &mut [(&mut Array<A, D>, &ArrayBase<S, D>)]) {
        for (param, grad) in params {
            let tmp = *grad * self.lr.clone();
            let new = &**param - &tmp;
            **param = new;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mat;

    #[test]
    fn test_sgd_update() {
        let mut sgd = Sgd { lr: 0.1 };

        let mut param = Mat::from_elem([2, 2], 1.0);
        let grad = Mat::from_elem([2, 2], 0.5);
        sgd.update(&mut [(&mut param, &grad)]);
        assert_eq!(param, Mat::from_elem([2, 2], 0.95));
    }
}
