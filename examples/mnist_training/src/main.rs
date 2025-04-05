extern crate openblas_src;

use std::collections::HashMap;

use alg::{
    math::{Sigmoid, Softmax},
    tensor::safetensors::Load,
};
use egui::{ColorImage, Image};
use ndarray::{Array1, Array2, Axis};

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

fn predict(network: &HashMap<String, Array2<f32>>, x: &Array1<f32>) -> Array1<f32> {
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

fn main() {
    let mnist_dir = std::env::home_dir().unwrap().join("Work/mnist");

    let train_data_path = mnist_dir.join("train-images-idx3-ubyte");
    let data = alg::dataset::mnist::load_images(train_data_path);

    let network = init_network();

    eframe::run_native(
        "mnist",
        eframe::NativeOptions::default(),
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);

            Ok(Box::new(App {
                network,
                data,
                idx: 0,
            }))
        }),
    )
    .unwrap();
}

struct App {
    network: HashMap<String, Array2<f32>>,
    data: Array2<u8>,
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

            let img_data = self.data.index_axis(Axis(0), self.idx).to_vec();

            let img_data_a = self
                .data
                .index_axis(Axis(0), self.idx)
                .map(|a| *a as f32 / 255.0);

            let result = predict(&self.network, &img_data_a).to_vec();

            let image = ColorImage::from_gray([28, 28], &img_data);
            let texture = ctx.load_texture("abc", image, egui::TextureOptions::default());

            let img = Image::new(&texture).shrink_to_fit();
            ui.add_sized([300.0, 300.0], img);

            if ui.button("--->").clicked() {
                self.idx = (self.idx + 1) % data_0_len;
            }

            let mut res: Vec<_> = (0..=9).into_iter().zip(result.into_iter()).collect();
            res.sort_by(|a, b| b.1.total_cmp(&a.1));

            ui.label(format!("predicted label: {:?}", res));
        });
    }
}
