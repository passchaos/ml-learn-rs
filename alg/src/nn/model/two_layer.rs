use vectra::prelude::{Array, Matmul};

use crate::{
    math::{ActivationFn, LossFn},
    nn::{matmul_policy, numerical_gradient},
};
use num::NumCast;

type Ft = f64;

use super::layer::{Affine, SoftmaxWithLoss};
type Mat = super::layer::Mat<Ft>;
//
use super::layer::{AffineN, Relu, SoftmaxWithLossN};
type MatN = ndarray::Array2<Ft>;

struct TwoLayerModel {
    // w1: Mat,
    // b1: Mat,
    // w2: Mat,
    // b2: Mat,
    // affine1: Affine<Ft>,
    // relu: Relu<Ft>,
    affine2: Affine<Ft>,
    last_layer: SoftmaxWithLoss<Ft>,
}

impl TwoLayerModel {
    pub fn new(w: Mat) -> Self {
        let (input_size, output_size) = (w.shape()[0], w.shape()[1]);

        // let w2 = w;
        // let w2 = MatN::from_shape_vec((input_size, output_size), w.data().to_vec()).unwrap();
        let b2 = Mat::zeros([1, output_size]);

        // let relu = Relu::default();
        let affine2 = Affine::new(w, b2);

        TwoLayerModel {
            // w1,
            // b1,
            // w2,
            // b2,
            // affine1,
            // relu,
            affine2,
            last_layer: SoftmaxWithLoss::default(),
        }
    }

    pub fn predict(&mut self, mut x: Mat) -> Mat {
        // x = self.affine1.forward(x);

        // if x.contains_nan() {
        //     println!("x: {x:?}");
        //     std::process::exit(0);
        // }
        // x = self.relu.forward(x);
        // println!("affine2 w: {:?} x: {:?}", self.affine2.w, x);

        let w1 =
            MatN::from_shape_vec(self.affine2.w.shape(), self.affine2.w.data().to_vec()).unwrap();
        let x1 = MatN::from_shape_vec(x.shape(), x.data().to_vec()).unwrap();
        let res1 = x1.dot(&w1);

        x = self.affine2.forward(x);

        let res = Mat::from_vec(res1.into_raw_vec_and_offset().0, x.shape());

        let diff = (&x - &res).abs().sum();
        println!("predict diff: {diff}");

        // if x.contains_nan() {
        //     println!("x2: {x:?}");
        //     std::process::exit(0);
        // }

        x
        // let h = &x.matmul(&self.w1, matmul_policy()) + &self.b1;
        // let a = h.sigmoid();
        // let y = &a.matmul(&self.w2, matmul_policy()) + &self.b2;

        // y.softmax()
    }

    pub fn loss(&mut self, x: Mat, t: Mat) -> Ft {
        let y = self.predict(x);

        // println!("predict res: {y:?}");
        let loss = self.last_layer.forward(y, t);

        loss
    }

    // pub fn accuracy(&mut self, x: MatN, t: &MatN) -> Ft {
    //     let batch_size = x.shape()[0];

    //     let y = self.predict(x);

    //     let y = y.argmax_axis(1);
    //     let t = t.argmax_axis(1);

    //     let equal_count = y.equal(&t).into_map(|v| if v { 1 } else { 0 }).sum();

    //     <Ft as num::NumCast>::from(equal_count).unwrap()
    //         / <Ft as num::NumCast>::from(batch_size).unwrap()
    // }

    pub fn gradient(&mut self, x: Mat, t: &Mat) -> (f64, Mat, Mat, Mat) {
        // must invoked to clear numerical gradient cache
        let loss = self.loss(x, t.clone());
        // println!("gradient inner loss= {loss}");

        let mut grad = self.last_layer.backward();

        // if grad.contains_nan() {
        //     println!("last layer grad: {grad:?}");

        //     std::process::exit(0);
        // }

        let grad2 = self.affine2.backward(grad.clone());

        // if grad2.contains_nan() {
        //     println!("affine2 grad: {grad2:?}");
        //     std::process::exit(0);
        // }

        // prinln!("grad2: {grad2:?}");
        // grad = self.relu.backward(grad2.clone());
        // let grad1 = self.affine1.backward(grad2);

        // if grad1.contains_nan() {
        //     println!("affine1 grad: {grad1:?}");

        //     std::process::exit(0);
        // }

        // let dw1 = Array::zeros(self.affine1.w.shape());
        // let db1 = Array::zeros(self.affine1.b.shape());
        // let dw1 = self.affine1.dw.clone().unwrap();
        // let db1 = self.affine1.db.clone().unwrap();
        let dw2 = self.affine2.dw.clone().unwrap();
        // let db2 = self.affine2.db.clone().unwrap();
        // let db2 = MatN::zeros(self.affine2.b.shape());

        (loss, grad, grad2, dw2)
        // (dw1, db1, dw2, db2)
    }

    // pub fn numerical_gradient(&mut self, x: MatN, t: &MatN) -> MatN {
    //     // let w1 = self.affine1.w.clone();
    //     // let b1 = self.affine1.b.clone();
    //     let w2 = self.affine2.w.clone();
    //     let b2 = self.affine2.b.clone();

    //     // let loss_w1 = |w: &Mat| {
    //     //     self.affine1.w = w.clone();
    //     //     let loss = self.loss(x.clone(), t.clone());

    //     //     loss
    //     // };

    //     // let (dw1, _) = numerical_gradient(loss_w1, w1);
    //     // // println!("dw1: {dw1:?}");

    //     // let loss_b1 = |b: &Mat| {
    //     //     self.affine1.b = b.clone();
    //     //     let loss = self.loss(x.clone(), t.clone());

    //     //     loss
    //     // };

    //     // let (db1, _) = numerical_gradient(loss_b1, b1);

    //     let loss_o = self.loss(x.clone(), t.clone());
    //     // println!("loss_o= {loss_o}");

    //     let w2_c = w2.clone();
    //     let loss_w2 = |w: &MatN| {
    //         let diff = (&w2_c - w).abs().sum();
    //         // println!("diff: {diff}");

    //         self.affine2.w = w.clone();
    //         let loss = self.loss(x.clone(), t.clone());

    //         // if loss != loss_o {
    //         //     println!("loss: {loss}");
    //         // }

    //         loss
    //     };

    //     // let (dw2, _) = numerical_gradient(loss_w2, w2);

    //     // let loss_b2 = |b: &Mat| {
    //     //     self.affine2.b = b.clone();
    //     //     let loss = self.loss(x.clone(), t.clone());

    //     //     loss
    //     // };

    //     // let (db2, _) = numerical_gradient(loss_b2, b2);

    //     x
    //     // (dw1, db1, dw2, db2)
    // }
}

struct TwoLayerModelN {
    // w1: Mat,
    // b1: Mat,
    // w2: Mat,
    // b2: Mat,
    // affine1: Affine<Ft>,
    // relu: Relu<Ft>,
    affine2: AffineN<Ft>,
    last_layer: SoftmaxWithLossN<Ft>,
}

impl TwoLayerModelN {
    pub fn new(w: MatN) -> Self {
        let (input_size, output_size) = (w.shape()[0], w.shape()[1]);

        // let w2 = w;
        // let w2 = MatN::from_shape_vec((input_size, output_size), w.data().to_vec()).unwrap();
        let b2 = MatN::zeros([1, output_size]);

        // let relu = Relu::default();
        let affine2 = AffineN::new(w, b2);

        Self {
            // w1,
            // b1,
            // w2,
            // b2,
            // affine1,
            // relu,
            affine2,
            last_layer: SoftmaxWithLossN::default(),
        }
    }

    pub fn predict(&mut self, mut x: MatN) -> MatN {
        // x = self.affine1.forward(x);

        // if x.contains_nan() {
        //     println!("x: {x:?}");
        //     std::process::exit(0);
        // }
        // x = self.relu.forward(x);

        // println!("affine2 n w: {:?} x: {:?}", self.affine2.w, x);
        x = self.affine2.forward(x);

        // if x.contains_nan() {
        //     println!("x2: {x:?}");
        //     std::process::exit(0);
        // }

        x
        // let h = &x.matmul(&self.w1, matmul_policy()) + &self.b1;
        // let a = h.sigmoid();
        // let y = &a.matmul(&self.w2, matmul_policy()) + &self.b2;

        // y.softmax()
    }

    pub fn loss(&mut self, x: MatN, t: MatN) -> Ft {
        let y = self.predict(x);

        // println!("predict res n: {y:?}");
        let loss = self.last_layer.forward(y, t);

        loss
    }

    // pub fn accuracy(&mut self, x: MatN, t: &MatN) -> Ft {
    //     let batch_size = x.shape()[0];

    //     let y = self.predict(x);

    //     let y = y.argmax_axis(1);
    //     let t = t.argmax_axis(1);

    //     let equal_count = y.equal(&t).into_map(|v| if v { 1 } else { 0 }).sum();

    //     <Ft as num::NumCast>::from(equal_count).unwrap()
    //         / <Ft as num::NumCast>::from(batch_size).unwrap()
    // }

    pub fn gradient(&mut self, x: MatN, t: &MatN) -> (f64, MatN, MatN, MatN) {
        // must invoked to clear numerical gradient cache
        let loss = self.loss(x, t.clone());
        // println!("gradient inner loss= {loss}");

        let mut grad = self.last_layer.backward();

        // if grad.contains_nan() {
        //     println!("last layer grad: {grad:?}");

        //     std::process::exit(0);
        // }

        let grad2 = self.affine2.backward(grad.clone());

        // if grad2.contains_nan() {
        //     println!("affine2 grad: {grad2:?}");
        //     std::process::exit(0);
        // }

        // prinln!("grad2: {grad2:?}");
        // grad = self.relu.backward(grad2.clone());
        // let grad1 = self.affine1.backward(grad2);

        // if grad1.contains_nan() {
        //     println!("affine1 grad: {grad1:?}");

        //     std::process::exit(0);
        // }

        // let dw1 = Array::zeros(self.affine1.w.shape());
        // let db1 = Array::zeros(self.affine1.b.shape());
        // let dw1 = self.affine1.dw.clone().unwrap();
        // let db1 = self.affine1.db.clone().unwrap();
        let dw2 = self.affine2.dw.clone().unwrap();
        // let db2 = self.affine2.db.clone().unwrap();
        // let db2 = MatN::zeros(self.affine2.b.shape());

        (loss, grad, grad2, dw2)
        // (dw1, db1, dw2, db2)
    }

    pub fn numerical_gradient(&mut self, x: MatN, t: &MatN) -> MatN {
        // let w1 = self.affine1.w.clone();
        // let b1 = self.affine1.b.clone();
        let w2 = self.affine2.w.clone();
        let b2 = self.affine2.b.clone();

        // let loss_w1 = |w: &Mat| {
        //     self.affine1.w = w.clone();
        //     let loss = self.loss(x.clone(), t.clone());

        //     loss
        // };

        // let (dw1, _) = numerical_gradient(loss_w1, w1);
        // // println!("dw1: {dw1:?}");

        // let loss_b1 = |b: &Mat| {
        //     self.affine1.b = b.clone();
        //     let loss = self.loss(x.clone(), t.clone());

        //     loss
        // };

        // let (db1, _) = numerical_gradient(loss_b1, b1);

        let loss_o = self.loss(x.clone(), t.clone());
        // println!("loss_o= {loss_o}");

        let w2_c = w2.clone();
        let loss_w2 = |w: &MatN| {
            let diff = (&w2_c - w).abs().sum();
            // println!("diff: {diff}");

            self.affine2.w = w.clone();
            let loss = self.loss(x.clone(), t.clone());

            // if loss != loss_o {
            //     println!("loss: {loss}");
            // }

            loss
        };

        // let (dw2, _) = numerical_gradient(loss_w2, w2);

        // let loss_b2 = |b: &Mat| {
        //     self.affine2.b = b.clone();
        //     let loss = self.loss(x.clone(), t.clone());

        //     loss
        // };

        // let (db2, _) = numerical_gradient(loss_b2, b2);

        x
        // (dw1, db1, dw2, db2)
    }
}

#[cfg(test)]
mod tests {
    use rand::seq::IndexedRandom;
    use vectra::prelude::Array;

    use crate::dataset::mnist::load_mnist;

    use super::*;

    #[test]
    fn test_reverse_propagation() {
        let x = Array::from_vec(vec![1.0, -0.5, -2.0, 3.0], [2, 2]);
    }

    #[test]
    fn test_loss() {
        // let mut model = TwoLayerModel::new(784, 50, 10);

        // let x = Mat::ones([100, 784]);
        // let t = Mat::ones([100, 10]);

        // let grads = model.numerical_gradient(x, &t);
        // println!("grads: {grads:?}");
    }

    fn weight_datas() -> Vec<Ft> {
        let v = vec![
            1.3041755931268144,
            1.3098438226867162,
            -0.3300688097279237,
            0.983000365378336,
            -0.6365619887814759,
            0.9486019450107478,
            0.36937716663300724,
            2.056399225801065,
            0.2124292373087392,
            -1.3056577776830811,
            0.559919823990334,
            -0.40223244325342855,
            0.6614721976372704,
            0.5154438666410969,
            0.950524324803184,
            0.27414211124396776,
            0.4370177164127968,
            0.7157718229745096,
            0.846451364587079,
            0.13113478026403338,
            0.24064974751760834,
            0.890667300688533,
            -1.884566475351754,
            -0.410353278350795,
            0.7762848517883381,
            2.558011731561468,
            -0.30976290917408106,
            0.10206657616799736,
            -1.649653329326057,
            0.49554811574796914,
            -0.7341004628430748,
            -1.2673911999310088,
            -0.9417676866786587,
            -0.6879995296286927,
            -0.02357062894543683,
            -0.6671986018017854,
            0.4094556737139499,
            0.39630072419171775,
            0.4627366199450396,
            0.5284846508125467,
            0.17706869376040388,
            0.7395043994201587,
            2.6215370529648045,
            -1.7219017200977724,
            0.3331101613064511,
            1.7277494437281253,
            0.38638905927955125,
            0.6304344447999607,
            -1.305634048483957,
            0.09066866232508623,
            -0.8902728460374478,
            -1.0136781277411608,
            -0.9172559399351445,
            -0.7037740821316988,
            0.45477911972014484,
            -0.33576695058319356,
            -0.5603523099597643,
            -1.500713873628056,
            0.5456066165824106,
            1.0808805274632425,
            0.17975001348027456,
            -0.9301151378963065,
            -0.7509934529971979,
            -0.14848132245411585,
            -0.4378386123611622,
            0.09958283380418483,
            0.837445415382545,
            0.3732531631366856,
            1.2990737890828912,
            0.5171094574530131,
            -0.3109226845286774,
            0.14720278647532362,
            -0.08852597039102247,
            1.306995925497403,
            -2.597101144871634,
            -0.9243472667571617,
            0.417657444591854,
            -0.3387463514653905,
            0.9362662687780139,
            0.9742207326818211,
            -2.789739609605662,
            -0.3091238055966574,
            0.14462372014551417,
            -1.1010771088267197,
            0.41314863570728444,
            -0.3655139585493735,
            0.2558550779601474,
            0.6951055770671437,
            1.4061173978758272,
            0.7126248198228382,
            0.7054260660997232,
            1.2449308189706632,
            1.3435758622366942,
            0.4171794215320668,
            -0.5059436463498662,
            0.6218181033676955,
            -0.09725747114920538,
            -0.2374567410202262,
            0.5103922489709589,
            -0.42085520184537145,
            -0.2573861733080357,
            1.2318310108701058,
            -0.6900274300012514,
            -0.29153562435020514,
            -0.5020603334773687,
            -0.8660943270900876,
            1.8453577665136798,
            0.7934605987001193,
            -0.901746543898098,
            -0.08746754192449438,
            -0.018352342082580873,
            -1.1466723916130446,
            0.8844962285536031,
            1.2421185147601124,
            1.0895731399617505,
            0.9563356503992382,
            1.7693259967451853,
            -0.9270639015679898,
            0.5022628937596109,
            0.2997074710211515,
            0.7039312147106032,
            0.42852970722090444,
            1.1918948894349572,
            -0.8071661117525694,
            1.363489783246572,
            -1.4468693577459757,
            0.6443840052055751,
            -0.6681625926338289,
            -1.0059130547449622,
            1.216801595199022,
            -1.2340357685172858,
            0.29482256073158586,
            -0.3031087739161725,
            -0.2625072214525679,
            -0.25830744209646334,
            0.4946826388017433,
            -0.6873708507094153,
            0.39664311827473064,
            -2.3471945623921586,
            -1.0179630462304925,
            0.13616142420506205,
            0.9787580236336826,
            2.0811199616616043,
            -0.21301831065600627,
            -1.2267507736204755,
            -0.12926977673443055,
            0.40093351430299984,
            -0.9021673988523689,
            -0.1179808914599773,
            -0.5459151209049302,
            1.0702856730017394,
            -0.4493017553831248,
            0.11494651472004361,
            1.510268943072078,
            -0.7978320839091698,
            -0.19736179291261166,
            1.4329849862802448,
            1.5571235241840895,
            -0.45009337409984596,
            -0.8754010910928388,
            0.35763497205889955,
            -0.24151308216923423,
            0.1906795256298495,
            0.7592745187665777,
            1.7947044457987211,
            -0.03014868089013662,
            -0.44338256701238693,
            -1.0836970410266669,
            -0.5437038602707679,
            0.17579456648867708,
            1.5013886079363012,
            -0.13411898393984362,
            0.5993822941740833,
            -0.3370388390934838,
            0.504469493808434,
            0.3792178355319999,
            1.5914374015230992,
            -1.2247946382768735,
            -1.0589225496606327,
            -0.6739469463115889,
            0.1968678876693607,
            0.22281858868462032,
            -1.3433667732090342,
            -1.0877962317677892,
            0.8433646203465026,
            -1.9232017895684406,
            0.4246987939231476,
            0.1960611780219627,
            -0.21178863502561998,
            0.7572977938001425,
            -0.2504846504793454,
            -0.6756853486752029,
            -0.5351777282006928,
            0.9067043122804556,
            0.5236993097392344,
            0.3315296393906866,
            -1.8792359831695706,
            1.7375164958784663,
            0.2403733247888052,
            0.3328973640223677,
        ];
        v
    }

    #[test]
    fn test_mln_batch_loss() {
        let ((x_train, t_train), (x_test, t_test)) = load_mnist();

        let iters_num = 100;
        let train_size = x_train.shape()[0];

        let batch_size = 50;
        let learning_rate = 0.1;

        let mut rng = rand::rng();

        let samples: Vec<_> = (0..train_size).collect();

        // let begin = 100;

        let mut indices: Vec<_> = (229..239).collect();
        indices.extend_from_slice(&(458..468).collect::<Vec<_>>());

        // let w = Array::randn([indices.len(), 10]);
        let w = weight_datas();

        // let s = w
        //     .multi_iter()
        //     .fold(String::new(), |acc, (_, v)| acc + &format!("{v}, "));
        println!("w: {}", w.len());

        // let w = MatN::from_shape_vec([20, 10], w).unwrap();
        let mut model = TwoLayerModel::new(Mat::from_vec(w.clone(), [20, 10]));

        let mut model_n = TwoLayerModelN::new(MatN::from_shape_vec([20, 10], w).unwrap());
        // let indices: Vec<_> = (begin..(begin + input_size)).collect();
        //

        for i in 0..1 {
            let batch_mask = vec![1, 2];
            // let batch_mask: Vec<_> = samples
            //     .choose_multiple(&mut rng, batch_size)
            //     .cloned()
            //     .collect();

            let x_train_batch = x_train.select(0, &batch_mask);
            let t_train_batch = t_train.select(0, &batch_mask);

            let x_train_batch = x_train_batch.select(1, &indices);
            // println!("x_train: {x_train_batch:?}");
            // println!("t_train: {t_train_batch:?}");
            // let t_train_batch = t_train_batch.select(1, &indices);

            // let (dw1, db1, dw2, db2) =
            //     model.numerical_gradient(x_train_batch.clone(), &t_train_batch);

            // let (dw1, db1, dw2, db2) = model.gradient(x_train_batch.clone(), &t_train_batch);
            // let (dw1_c, db1_c, dw2_c, db2_c) =
            //     model.gradient(x_train_batch.clone(), &t_train_batch);
            // println!("dw2: {dw2:?} dw2_c: {dw2_c:?}");
            //
            let x_train_batch_n = MatN::from_shape_vec(
                (x_train_batch.shape()[0], x_train_batch.shape()[1]),
                x_train_batch.data().to_vec(),
            )
            .unwrap();
            let t_train_batch_n = MatN::from_shape_vec(
                (t_train_batch.shape()[0], t_train_batch.shape()[1]),
                t_train_batch.data().to_vec(),
            )
            .unwrap();

            fn diff_vectra_ndarray(x1: Mat, x2: MatN) -> Ft {
                let x1_n = Mat::from_vec(x2.into_raw_vec_and_offset().0, x1.shape());
                (&x1 - &x1_n).abs().sum()
            }

            let (loss, grad, grad2, dw2) = model.gradient(x_train_batch.clone(), &t_train_batch);
            let (loss_n, grad_n, grad2_n, dw2_n) =
                model_n.gradient(x_train_batch_n.clone(), &t_train_batch_n);

            let diff_grad = diff_vectra_ndarray(grad, grad_n);
            let diff_grad2 = diff_vectra_ndarray(grad2, grad2_n);
            let diff_dw2 = diff_vectra_ndarray(dw2.clone(), dw2_n.clone());
            println!("diff: grad= {diff_grad} grad2= {diff_grad2} dw2= {diff_dw2}");

            // let diff_w1 = (&dw1 - &dw1_c).abs().sum();
            // let diff_b1 = (&db1 - &db1_c).abs().sum();
            // let diff_w2 = (&dw2 - &dw2_c).abs().sum();
            // let diff_b2 = (&db2 - &db2_c).abs().sum();
            // println!("diff_w1: {diff_w1} diff_b1: {diff_b1} diff_w2: {diff_w2} diff_b2: {diff_b2}");

            // let diff = diff_w1 + diff_b1 + diff_w2 + diff_b2;
            // if diff > 10.0 {
            //     std::process::exit(0);
            // }

            // model.affine1.w -= dw1.mul_scalar(learning_rate);
            // model.affine1.b -= db1.mul_scalar(learning_rate);
            model.affine2.w -= dw2.mul_scalar(learning_rate);
            model_n.affine2.w -= &(dw2_n * learning_rate);
            // model.affine2.b -= db2.mul_scalar(learning_rate);

            let loss = model.loss(x_train_batch.clone(), t_train_batch.clone());
            let loss_n = model_n.loss(x_train_batch_n, t_train_batch_n);
            println!("i= {i} loss= {loss} loss_n= {loss_n}");
        }
    }
}
