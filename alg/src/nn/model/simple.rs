use vectra::prelude::*;

use crate::{
    math::{ActivationFn, LossFn},
    nn::{matmul_policy, numerical_gradient},
};

type Ft = f64;

type Mat = super::layer::Mat<Ft>;

use super::layer::{Affine, Relu, SoftmaxWithLoss};

struct SimpleNet {
    // w: Mat,
    // b: Mat,
    affine: Affine<Ft>,
    last_layer: SoftmaxWithLoss<Ft>,
}

impl SimpleNet {
    pub fn new(w: Mat, b: Mat) -> Self {
        let affine = Affine::new(w, b);
        let last_layer = SoftmaxWithLoss::default();

        SimpleNet {
            // w,
            // b,
            affine,
            last_layer,
        }
    }

    pub fn predict(&mut self, x: Mat) -> Mat {
        // println!("w= {:?} affine_w= {:?}", self.w, self.affine.w);
        let v1 = self.affine.forward(x.clone());
        // let v2 = &(x.matmul(&self.w, matmul_policy())) + &self.b;

        // let diff = &(&v1 - &v2).abs().sum();
        // println!("predict diff: {diff}");
        v1
    }

    pub fn loss(&mut self, x: Mat, t: Mat) -> Ft {
        let y = self.predict(x);

        self.last_layer.forward(y, t)
    }

    pub fn gradient(&mut self) -> (Mat, Mat) {
        let grad = self.last_layer.backward();
        let dx = self.affine.backward(grad);

        (
            self.affine.dw.clone().unwrap(),
            self.affine.db.clone().unwrap(),
        )
    }

    pub fn numerical_gradient(&mut self, x: Mat, t: &Mat) -> (Mat, Mat) {
        let w_clone = self.affine.w.clone();
        let simple_w_f = |w: &Mat| {
            // std::mem::swap(&mut model.w, w);
            self.affine.w = w.clone();
            let loss = self.loss(x.clone(), t.clone());
            // println!("w_f: loss= {loss} w= {w:?}");

            loss
        };
        let (dw, _) = numerical_gradient(simple_w_f, w_clone);

        let b_clone = self.affine.b.clone();
        let simple_b_f = |b: &Mat| {
            // std::mem::swap(&mut model.w, w);
            self.affine.b = b.clone();
            let loss = self.loss(x.clone(), t.clone());

            loss
        };
        let (db, _) = numerical_gradient(simple_b_f, b_clone);

        (dw, db)
    }
}
#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use vectra::prelude::Array;

    use crate::nn::numerical_gradient;

    use super::*;

    #[test]
    fn test_simple() {
        let w = Array::from_vec(
            vec![
                0.47355232, 0.9977393, 0.84668094, 0.85557411, 0.03563661, 0.69422093,
            ],
            [2, 3],
        );
        let b = Array::zeros([1, 3]);
        let mut model = SimpleNet::new(w, b);

        let x = Array::from_vec(vec![0.6, 0.9], [1, 2]);
        let p = model.predict(x.clone());
        println!("p: {p:?}");
        assert_relative_eq!(
            p,
            Array::from_vec(vec![1.05414809, 0.63071653, 1.1328074], [1, 3]),
            epsilon = 1e-6
        );

        let idx = p.argmax();

        println!("idx: {idx:?}");
        assert_eq!(idx, [0, 2]);

        let t = Array::from_vec(vec![0.0, 0.0, 1.0], [1, 3]);
        let loss = model.loss(x.clone(), t.clone());
        println!("loss: {loss}");
        assert_relative_eq!(loss, 0.92806853663411326, epsilon = 1e-6);
    }

    #[test]
    fn test_simple_gradient() {
        let w = Array::from_vec(
            vec![
                0.47355232, 0.9977393, 0.84668094, 0.85557411, 0.03563661, 0.69422093,
            ],
            [2, 3],
        );
        let b = Array::zeros([1, 3]);
        let mut model = SimpleNet::new(w, b);

        let x = Array::from_vec(vec![0.6, 0.9], [1, 2]);
        let t = Array::from_vec(vec![0.0, 0.0, 1.0], [1, 3]);

        let loss = model.loss(x.clone(), t.clone());

        println!(
            "begin gradient handle: loss= {loss}================================================="
        );

        let grad1 = model.gradient();
        println!("grad1: {grad1:?}");
        let grad2 = model.numerical_gradient(x.clone(), &t);
        println!("grad2= {grad2:?}");

        assert_relative_eq!(
            grad1.0,
            Array::from_vec(
                vec![
                    0.21924763,
                    0.14356247,
                    -0.36281009,
                    0.32887144,
                    0.2153437,
                    -0.54421514
                ],
                [2, 3]
            ),
            epsilon = 1e-6
        );

        for i in 0..1000 {
            let (dw, db) = model.gradient();
            let (dw1, db1) = model.numerical_gradient(x.clone(), &t);

            let dw_diff = (&dw - &dw1).abs().sum();
            let db_diff = (&db - &db1).abs().sum();
            println!("diff: dw= {dw_diff} db= {db_diff}");

            model.affine.w -= dw.mul_scalar(0.1);
            model.affine.b -= db.mul_scalar(0.1);

            let loss = model.loss(x.clone(), t.clone());
            println!("i= {i} loss= {loss}");
        }
    }
}
