extern crate openblas_src;

use std::collections::HashMap;

use alg::{
    math::{
        DigitalRecognition, Sigmoid, Softmax, loss::cross_entropy_error,
        normalize::NormalizeTransform, one_hot::OneHotTransform,
    },
    tensor::safetensors::Load,
};
use egui::{ColorImage, Image};
use ndarray::{Array1, Array2, ArrayView1, Axis, array};
use rand::Rng;

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
    use alg::math::autodiff::numerical_gradient;
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

        let f1 = |w: Array2<f32>| {
            let net = SimpleNet::new(w);
            net.loss(&x, t.clone())
        };

        let a = numerical_gradient(f1, w, 0.001);
        println!("a: {a}");
    }
}
