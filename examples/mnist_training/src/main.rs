extern crate openblas_src;

use std::collections::HashMap;

use alg::{
    layer::{affine::AffineLayer, relu::ReluLayer, softmax_loss::SoftmaxWithLossLayer},
    math::{
        DigitalRecognition, Relu, Sigmoid, Softmax, autodiff::numerical_gradient,
        loss::cross_entropy_error, normalize::NormalizeTransform, one_hot::OneHotTransform,
    },
    tensor::safetensors::Load,
};
use approx::relative_eq;
use egui::{ColorImage, Image};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Ix2, array, s};
use rand::Rng;

struct TwoLayerNetN {
    inner: TwoLayerNet,
    affine1_layer: AffineLayer<f32>,
    relu1_layer: ReluLayer<Ix2>,
    affine2_layer: AffineLayer<f32>,
    last_layer: SoftmaxWithLossLayer<f32, Ix2>,
}

impl From<TwoLayerNet> for TwoLayerNetN {
    fn from(inner: TwoLayerNet) -> Self {
        let affine1_layer = AffineLayer::new(inner.w1.clone(), inner.b1.clone());
        let relu1_layer = ReluLayer::default();
        let affine2_layer = AffineLayer::new(inner.w2.clone(), inner.b2.clone());
        let last_layer = SoftmaxWithLossLayer::default();

        Self {
            inner,
            affine1_layer,
            relu1_layer,
            affine2_layer,
            last_layer,
        }
    }
}

impl TwoLayerNetN {
    fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        weight_init_std: f32,
    ) -> Self {
        let inner = TwoLayerNet::new(input_size, hidden_size, output_size, weight_init_std);
        inner.into()
    }

    fn predict(&mut self, x: &ArrayView2<f32>) -> Array2<f32> {
        let a1 = self.affine1_layer.forward(x);
        let z1 = self.relu1_layer.forward(&a1.view());
        let a2 = self.affine2_layer.forward(&z1.view());

        a2
    }

    fn loss(&mut self, x: &ArrayView2<f32>, t: &Array2<f32>) -> f32 {
        let y = self.predict(x);
        self.last_layer.forward(&y, t)
    }

    fn numerical_gradient(&mut self, x: &ArrayView2<f32>, t: &Array2<f32>) -> Self {
        self.inner.numerical_gradient(x, t).into()
    }

    fn gradient(&mut self, x: &ArrayView2<f32>, t: &Array2<f32>) -> Self {
        self.loss(x, t);

        let dout = self.last_layer.backward();
        let (dout2, dw2, db2) = self.affine2_layer.backward(&dout.view());
        let dout = self.relu1_layer.backward(&dout2.view());
        let (dout1, dw1, db1) = self.affine1_layer.backward(&dout.view());

        let inner = TwoLayerNet {
            w1: dw1,
            b1: db1,
            w2: dw2,
            b2: db2,
        };

        inner.into()
    }
}

#[derive(Clone, Debug)]
struct TwoLayerNet {
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
}

impl TwoLayerNet {
    fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        weight_init_std: f32,
    ) -> Self {
        let w1 = alg::math::stat::randn((input_size, hidden_size)) * weight_init_std;
        let b1 = Array1::zeros(hidden_size);
        let w2 = alg::math::stat::randn((hidden_size, output_size)) * weight_init_std;
        let b2 = Array1::zeros(output_size);

        Self { w1, b1, w2, b2 }
    }

    fn predict(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let a1 = x.dot(&self.w1) + &self.b1;
        // let z1 = a1.sigmoid();
        let z1 = a1.relu();
        let a2 = z1.dot(&self.w2) + &self.b2;
        let y = a2.softmax();

        y
    }

    fn loss(&self, x: &ArrayView2<f32>, t: &Array2<f32>) -> f32 {
        let y = self.predict(x);
        cross_entropy_error(y, t.clone())
    }

    fn accuracy(&self, x: &ArrayView2<f32>, t: &Array2<f32>) -> f32 {
        let mut y = self.predict(x);
        let mut t = t.clone();

        let y1 = y.map_axis_mut(Axis(1), |arr| {
            arr.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx)
                .unwrap()
        });
        let t1 = t.map_axis_mut(Axis(1), |arr| {
            arr.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx)
                .unwrap()
        });

        let accuracy = y1
            .iter()
            .zip(t1.iter())
            .map(|(y_i, t_i)| if y_i == t_i { 1.0 } else { 0.0 })
            .sum::<f32>()
            / y1.len() as f32;

        accuracy
    }

    fn numerical_gradient(&mut self, x: &ArrayView2<f32>, t: &Array2<f32>) -> Self {
        let mut w1 = self.w1.clone();
        let mut b1 = self.b1.clone();
        let mut w2 = self.w2.clone();
        let mut b2 = self.b2.clone();

        let g_w1 = numerical_gradient(
            |w1: &Array2<f32>| {
                self.w1 = w1.clone();
                self.loss(x, t)
            },
            &mut w1,
            0.01,
        );

        let g_b1 = numerical_gradient(
            |b1: &Array1<f32>| {
                self.b1 = b1.clone();
                self.loss(x, t)
            },
            &mut b1,
            0.01,
        );

        let g_w2 = numerical_gradient(
            |w2: &Array2<f32>| {
                self.w2 = w2.clone();
                self.loss(x, t)
            },
            &mut w2,
            0.01,
        );

        let g_b2 = numerical_gradient(
            |b2: &Array1<f32>| {
                self.b2 = b2.clone();
                self.loss(x, t)
            },
            &mut b2,
            0.01,
        );

        Self {
            w1: g_w1,
            b1: g_b1,
            w2: g_w2,
            b2: g_b2,
        }
    }
}

#[derive(Debug, Clone)]
struct SimpleNet {
    w: Array2<f32>,
}

impl SimpleNet {
    fn new(w: Array2<f32>) -> Self {
        // let mut w = Array2::zeros((2, 3));
        // w.mapv_inplace(|a| rand::thread_rng().gen_range(0.0..1.0));

        Self { w }
    }

    fn predict(&self, x: &Array1<f32>) -> Array1<f32> {
        x.dot(&self.w)
    }

    fn loss(&self, x: &Array1<f32>, t: Array1<f32>) -> f32 {
        let z = self.predict(x);
        let y = z.softmax();

        let loss = cross_entropy_error(y, t);

        loss
    }
}

fn init_network() -> HashMap<String, Array2<f32>> {
    let file_path = std::env::home_dir()
        .unwrap()
        .join("Work/DeepLearningFromScratch/ch03/sample_weight.safetensors");
    let f = std::fs::File::open(file_path).unwrap();

    let buffer = unsafe { memmap2::MmapOptions::new().map_copy_read_only(&f).unwrap() };
    let tensors = safetensors::SafeTensors::deserialize(&buffer).unwrap();
    println!("tensors: keys= {:?}", tensors.names());

    let w1 = tensors.tensor("W1").unwrap().load().unwrap();
    let w2 = tensors.tensor("W2").unwrap().load().unwrap();
    let w3 = tensors.tensor("W3").unwrap().load().unwrap();
    let b1 = tensors.tensor("b1").unwrap().load().unwrap();
    let b2 = tensors.tensor("b2").unwrap().load().unwrap();
    let b3 = tensors.tensor("b3").unwrap().load().unwrap();

    let mut map = HashMap::new();
    map.insert("w1".to_string(), w1);
    map.insert("w2".to_string(), w2);
    map.insert("w3".to_string(), w3);
    map.insert("b1".to_string(), b1);
    map.insert("b2".to_string(), b2);
    map.insert("b3".to_string(), b3);

    for (k, v) in &map {
        println!("k: {k} shape= {:?}", v.shape());
    }

    map
}

fn predict(network: &HashMap<String, Array2<f32>>, x: ArrayView1<f32>) -> Array1<f32> {
    let w1 = &network["w1"];
    let w2 = &network["w2"];
    let w3 = &network["w3"];

    let b1 = &network["b1"];
    let b1 = b1.to_shape(b1.shape()[0]).unwrap();
    let b2 = &network["b2"];
    let b2 = b2.to_shape(b2.shape()[0]).unwrap();
    let b3 = &network["b3"];
    let b3 = b3.to_shape(b3.shape()[0]).unwrap();

    let a1 = x.dot(w1) + b1;
    let z1 = a1.sigmoid();

    let a2 = z1.dot(w2) + b2;

    let z2 = a2.sigmoid();
    let a3 = z2.dot(w3) + b3;

    let y = a3.softmax();

    y
}

fn load_mnist() -> ((Array2<f32>, Array2<f32>), (Array2<f32>, Array2<f32>)) {
    let mnist_dir = std::env::home_dir().unwrap().join("Work/mnist");

    let train_data_path = mnist_dir.join("train-images-idx3-ubyte");
    let train_data = alg::dataset::mnist::load_images(train_data_path);
    let train_data = DigitalRecognition::normalize(&train_data);

    let label_data_path = mnist_dir.join("train-labels-idx1-ubyte");
    let train_labels = alg::dataset::mnist::load_labels(label_data_path);
    let train_labels = DigitalRecognition::one_hot(&train_labels);

    let test_data_path = mnist_dir.join("t10k-images-idx3-ubyte");
    let test_data = alg::dataset::mnist::load_images(test_data_path);
    let test_data = DigitalRecognition::normalize(&test_data);

    let label_data_path = mnist_dir.join("t10k-labels-idx1-ubyte");
    let test_labels = alg::dataset::mnist::load_labels(label_data_path);
    let test_labels = DigitalRecognition::one_hot(&test_labels);

    ((train_data, train_labels), (test_data, test_labels))
}

fn main() {
    let ((x_train, t_train), (x_test, t_test)) = load_mnist();

    println!(
        "{:?} {:?} {:?} {:?}",
        x_train.shape(),
        t_train.shape(),
        x_test.shape(),
        t_test.shape()
    );

    let iters_num = 20000;
    let train_size = x_train.shape()[0];
    let batch_size = 100;
    let learning_rate = 0.1;

    // let mut network = TwoLayerNet::new(784, 50, 10, 0.01);
    let mut network: TwoLayerNetN = TwoLayerNet::new(784, 50, 10, 0.1).into();

    let mut rng = rand::rng();

    for i in 0..iters_num {
        let begin = std::time::Instant::now();
        let idx = rand::seq::index::sample(&mut rng, train_size, batch_size).into_vec();

        let x_batch = x_train.select(Axis(0), &idx);
        let t_batch = t_train.select(Axis(0), &idx);
        // println!("x= {x_batch} t= {t_batch}");

        // let grad = network.numerical_gradient(&x_batch.view(), &t_batch);
        let grad = network.gradient(&x_batch.view(), &t_batch);

        // println!("w1: old= {} new= {}", grad_old.inner.w1, grad.inner.w1);

        // let va = relative_eq!(grad_old.inner.w1, grad.inner.w1);
        // if !va {
        //     return;
        // }

        let grad = grad.inner;
        let mut inner_network = network.inner.clone();
        // let mut inner_network = network.clone();
        // let mut inner_network = grad.inner.clone();
        inner_network.w1 = (inner_network.w1 - learning_rate * grad.w1);
        inner_network.b1 = (inner_network.b1 - learning_rate * grad.b1);
        inner_network.w2 = (inner_network.w2 - learning_rate * grad.w2);
        inner_network.b2 = (inner_network.b2 - learning_rate * grad.b2);

        let loss_inner = inner_network.loss(&x_batch.view(), &t_batch);

        network = inner_network.into();

        let loss = network.loss(&x_batch.view(), &t_batch);
        let elapsed = begin.elapsed();
        println!(
            "loss info: idx= {i} inner_value= {loss_inner} value= {loss} elapsed= {elapsed:?}"
        );

        if loss.is_nan() {
            return;
        }
    }
    return;

    let network = init_network();

    eframe::run_native(
        "mnist",
        eframe::NativeOptions::default(),
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);

            Ok(Box::new(App {
                network,
                data: x_train,
                labels: t_train,
                idx: 0,
            }))
        }),
    )
    .unwrap();
}

struct App {
    network: HashMap<String, Array2<f32>>,
    data: Array2<f32>,
    labels: Array2<f32>,
    idx: usize,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let data_0_len = self.data.len_of(Axis(0));

            ui.label(format!("current data row: {}", self.idx));

            if ui.button("<---").clicked() {
                self.idx = (self.idx + data_0_len - 1) % data_0_len;
            }

            let img_data = self
                .data
                .index_axis(Axis(0), self.idx)
                .map(|a| (a * 255.0) as u8)
                .to_vec();

            let img_data_a = self.data.index_axis(Axis(0), self.idx);

            let result = predict(&self.network, img_data_a).to_vec();

            let image = ColorImage::from_gray([28, 28], &img_data);
            let texture = ctx.load_texture("abc", image, egui::TextureOptions::default());

            let img = Image::new(&texture).shrink_to_fit();
            ui.add_sized([300.0, 300.0], img);

            if ui.button("--->").clicked() {
                self.idx = (self.idx + 1) % data_0_len;
            }

            let mut res: Vec<_> = (0..=9).into_iter().zip(result.into_iter()).collect();
            res.sort_by(|a, b| b.1.total_cmp(&a.1));

            let actual_label = self.labels.index_axis(Axis(0), self.idx);

            ui.label(format!(
                "actual label: {actual_label} predicted label: {res:?}"
            ));
        });
    }
}

#[cfg(test)]
mod tests {
    use std::sync::RwLock;

    use alg::math::{autodiff::numerical_gradient, stat::randn};
    use approx::assert_relative_eq;
    use ndarray::arr1;

    use super::*;

    #[test]
    fn test_simple_network() {
        let mut w = array![
            [0.47355232, 0.9977393, 0.84668094],
            [0.85557411, 0.03563661, 0.69422093]
        ];

        let net = SimpleNet::new(w.clone());

        let x = arr1(&[0.6, 0.9]);
        let p = net.predict(&x);

        // one-hot 表示
        let t = arr1(&[0.0, 0.0, 1.0]);
        println!("w: {net:?} p: {p} loss= {}", net.loss(&x, t.clone()));

        let f1 = |w: &Array2<f32>| {
            let net = SimpleNet::new(w.clone());
            net.loss(&x, t.clone())
        };

        let a = numerical_gradient(f1, &mut w, 0.001);

        assert_relative_eq!(
            a,
            array![
                [0.21924763, 0.14356247, -0.36281009],
                [0.32887144, 0.2153437, -0.54421514]
            ],
            max_relative = 0.001
        );
        println!("a: {a}");

        // test for SimpleNet reuse
        // let mut net = SimpleNet::new(w.clone());

        // let f1 = |w: &Array2<f32>| net.loss(&x, t.clone());

        // let a = numerical_gradient(f1, &mut net.w, 0.001);

        // assert_relative_eq!(
        //     a,
        //     array![
        //         [0.21924763, 0.14356247, -0.36281009],
        //         [0.32887144, 0.2153437, -0.54421514]
        //     ],
        //     max_relative = 0.001
        // );
        // println!("a: {a}");
    }

    #[test]
    fn test_two_layer_net() {
        let mut net = TwoLayerNet::new(784, 100, 10, 0.01);

        let x = randn((100, 784));
        let t = randn((100, 10));

        let grads = net.numerical_gradient(&x.view(), &t);
        println!("grads: {grads:?}");
        // let t = net.predict(&x);
    }
}
