extern crate blas_src;

use egui::mutex::RwLock;
// use rand::seq::IndexedRandom;
use rand::{Rng, prelude::IndexedRandom};

use std::{
    collections::{BTreeMap, HashMap},
    fmt::format,
    sync::Arc,
    time::Instant,
};

use alg::{
    math::{DigitalRecognition, normalize::NormalizeTransform, one_hot::OneHotTransform},
    nn::{
        layer::{
            Layer, LayerWard,
            dropout::Dropout,
            linear::{Linear, WeightInit},
            relu::Relu,
            softmax_loss::SoftmaxWithLoss,
        },
        model::Model,
        optimizer::{AdaGrad, Adam, Momentum, Optimizer, OptimizerOpT, Sgd},
    },
};
use egui_plot::{Legend, PlotPoints, Points};
use ndarray::{Array2, Axis, s};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn load_mnist() -> ((Array2<f32>, Array2<f32>), (Array2<f32>, Array2<f32>)) {
    let mnist_dir = std::env::home_dir().unwrap().join("Work/mnist");

    let train_data_path = mnist_dir.join("train-images.idx3-ubyte");
    let train_data = alg::dataset::mnist::load_images(train_data_path);
    let train_data = DigitalRecognition::normalize(&train_data);

    let label_data_path = mnist_dir.join("train-labels.idx1-ubyte");
    let train_labels = alg::dataset::mnist::load_labels(label_data_path);
    let train_labels = DigitalRecognition::one_hot(&train_labels);

    let test_data_path = mnist_dir.join("t10k-images.idx3-ubyte");
    let test_data = alg::dataset::mnist::load_images(test_data_path);
    let test_data = DigitalRecognition::normalize(&test_data);

    let label_data_path = mnist_dir.join("t10k-labels.idx1-ubyte");
    let test_labels = alg::dataset::mnist::load_labels(label_data_path);
    let test_labels = DigitalRecognition::one_hot(&test_labels);

    ((train_data, train_labels), (test_data, test_labels))
}

fn model_train<R: Rng>(
    losses_map: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    model_name: &str,
    x_train: &Array2<f32>,
    t_train: &Array2<f32>,
    x_test: &Array2<f32>,
    t_test: &Array2<f32>,
    iters_num: i32,
    batch_size: usize,
    sample: &[usize],
    rng: &mut R,
    weight_init: WeightInit,
    optimizer: Optimizer,
    batch_norm_momentum: Option<f32>,
    dropout_ratio: Option<f32>,
) {
    let mut model = Model::new(
        784,
        &[100, 100],
        10,
        weight_init,
        batch_norm_momentum,
        dropout_ratio,
        optimizer,
    );

    // train loop speed:
    // loss使用x_batch
    // release: 2.38s
    // codegen=1: 2.15s
    // mimalloc: 2.05s
    // remove println: 1.99s
    for i in 0..iters_num {
        let batch_mask: Vec<_> = sample.choose_multiple(rng, batch_size).cloned().collect();

        let x_batch = x_train.select(Axis(0), batch_mask.as_slice());
        let t_batch = t_train.select(Axis(0), batch_mask.as_slice());

        let loss = model.loss(&x_batch, &t_batch);
        model.backward();

        let loss = model.loss(x_test, t_test);
        // println!("elapsed: 1_1= {elapsed11} 1= {elapsed1}, 2= {elapsed2}, 3= {elapsed3}");

        println!("idx: {i} loss: {loss}");

        losses_map
            .write()
            .entry(model_name.to_string())
            .or_insert_with(|| vec![])
            .push(loss);
    }
}

fn train_logic(losses_map: Arc<RwLock<HashMap<String, Vec<f32>>>>) {
    let ((x_train, t_train), (x_test, t_test)) = load_mnist();

    println!(
        "{:?} {:?} {:?} {:?}",
        x_train.shape(),
        t_train.shape(),
        x_test.shape(),
        t_test.shape()
    );

    let iters_num = 5000;
    let batch_size = 100;

    // let x_train = x_train.slice(s![0..1000, ..]).to_owned();
    // let t_train = t_train.slice(s![0..1000, ..]).to_owned();

    let train_size = x_train.shape()[0];
    // let x_train = x_train.select(Axis(0), &[0..1000]);
    // let t_train = t_train.select(Axis(0), &[0..1000]);

    let iter_per_epoch = std::cmp::max(train_size / batch_size as usize, 1);

    let mut rng = rand::rng();

    // let mut losses_map = HashMap::new();

    let sample: Vec<_> = (0..train_size).into_iter().collect();

    let comb = vec![
        // (WeightInit::Std(0.01), Optimizer::Sgd(Sgd::new(0.01))),
        // (
        //     WeightInit::Std(0.01),
        //     Optimizer::Momentum(Momentum::new(0.01, 0.9)),
        // ),
        // (
        //     WeightInit::Std(0.01),
        //     Optimizer::AdaGrad(AdaGrad::new(0.01)),
        // ),
        // (
        //     WeightInit::Std(0.01),
        //     Optimizer::Adam(Adam::new(0.001, 0.9, 0.999)),
        // ),
        // (WeightInit::Xavier, Optimizer::Sgd(Sgd::new(0.01))),
        // (
        //     WeightInit::Xavier,
        //     Optimizer::Momentum(Momentum::new(0.01, 0.9)),
        // ),
        // (WeightInit::Xavier, Optimizer::AdaGrad(AdaGrad::new(0.01))),
        // (
        //     WeightInit::Xavier,
        //     Optimizer::Adam(Adam::new(0.001, 0.9, 0.999)),
        // ),
        // (WeightInit::He, Optimizer::Sgd(Sgd::new(0.01))),
        // (
        //     WeightInit::He,
        //     Optimizer::Momentum(Momentum::new(0.01, 0.9)),
        // ),
        // (WeightInit::He, Optimizer::AdaGrad(AdaGrad::new(0.01))),
        (
            WeightInit::He,
            Optimizer::Adam(Adam::new(0.001, 0.9, 0.999)),
            None,
            None,
        ),
        (
            WeightInit::He,
            Optimizer::Adam(Adam::new(0.001, 0.9, 0.999)),
            Some(0.9),
            None,
        ),
        (
            WeightInit::He,
            Optimizer::Adam(Adam::new(0.001, 0.9, 0.999)),
            None,
            Some(0.2),
        ),
        (
            WeightInit::He,
            Optimizer::Adam(Adam::new(0.001, 0.9, 0.999)),
            Some(0.9),
            Some(0.2),
        ),
    ];

    for (weight_init, optimizer, batch_norm_momentum, dropout_ratio) in comb {
        let begin = Instant::now();

        let model_name = format!(
            "Weight={weight_init:?}_Opt={optimizer:?}_BatchNorm={batch_norm_momentum:?}_Dropout={dropout_ratio:?}"
        );

        model_train(
            losses_map.clone(),
            &model_name,
            &x_train,
            &t_train,
            &x_test,
            &t_test,
            iters_num,
            batch_size,
            &sample,
            &mut rng,
            weight_init,
            optimizer.clone(),
            batch_norm_momentum,
            dropout_ratio,
        );

        let elapsed = begin.elapsed().as_secs_f32();
        println!("elapsed: {elapsed} model= {model_name}");
    }
}

fn main() {
    let losses_map = Arc::new(RwLock::new(HashMap::new()));
    let app = LossApp {
        losses_map: losses_map.clone(),
    };

    std::thread::spawn(move || {
        train_logic(losses_map);
    });

    eframe::run_native(
        "mnist",
        eframe::NativeOptions::default(),
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);

            Ok(Box::new(app))
        }),
    )
    .unwrap();
}

struct LossApp {
    losses_map: Arc<RwLock<HashMap<String, Vec<f32>>>>,
}

impl eframe::App for LossApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // println!("update");
        egui::CentralPanel::default().show(ctx, |ui| {
            let legend = Legend::default().position(egui_plot::Corner::RightTop);
            egui_plot::Plot::new(format!("plot"))
                .legend(legend)
                .show(ui, |plot_ui| {
                    let guard = self.losses_map.read();
                    // println!("len: {}", guard.len());

                    for (name, losses) in &*guard {
                        // println!("name: {name} losses_len= {}", losses.len());

                        let points = losses
                            .into_iter()
                            .enumerate()
                            .map(|(idx, loss)| [idx as f64, *loss as f64]);

                        let points = PlotPoints::from_iter(points);

                        plot_ui.points(Points::new(name, points));
                    }
                });
        });

        ctx.request_repaint_after_secs(0.1);
    }
}
