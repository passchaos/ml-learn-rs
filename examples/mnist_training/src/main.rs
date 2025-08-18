extern crate blas_src;

// use rand::seq::IndexedRandom;
use rand::{Rng, prelude::IndexedRandom};

use std::{collections::BTreeMap, fmt::format, time::Instant};

use alg::{
    math::{DigitalRecognition, normalize::NormalizeTransform, one_hot::OneHotTransform},
    nn::{
        layer::{
            Layer, LayerWard,
            linear::{Linear, WeightInit},
            relu::Relu,
            softmax_loss::SoftmaxWithLoss,
        },
        optimizer::{AdaGrad, Adam, Momentum, Optimizer, OptimizerOpT, Sgd},
    },
};
use egui_plot::{Legend, PlotPoints, Points};
use ndarray::{Array2, Axis};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

struct Model {
    layers: Vec<Layer>,
    out: SoftmaxWithLoss,
}

impl Model {
    fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        weight_init: WeightInit,
        opt: Optimizer,
    ) -> Self {
        let lin1 = Linear::new(
            weight_init,
            input_size,
            hidden_size,
            opt.clone(),
            Some(opt.clone()),
            true,
        );
        let lin2 = Linear::new(
            weight_init,
            hidden_size,
            output_size,
            opt.clone(),
            Some(opt),
            true,
        );
        let relu = Relu::default();

        let layers = vec![Layer::Linear(lin1), Layer::Relu(relu), Layer::Linear(lin2)];
        let out = SoftmaxWithLoss::default();

        Self { layers, out }
    }

    fn predict(&mut self, x: &Array2<f32>) -> Array2<f32> {
        let mut x = x.clone();
        for layer in &mut self.layers {
            x = layer.forward(&x);
        }

        x
    }

    fn loss(&mut self, x: &Array2<f32>, t: &Array2<f32>) -> f32 {
        let y = self.predict(x);
        self.out.forward(&y, t)
    }

    fn backward(&mut self) {
        let mut dout = self.out.backward();

        for layer in self.layers.iter_mut().rev() {
            dout = layer.backward(&dout);
        }
    }
}

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
) -> Vec<f32> {
    let mut model = Model::new(784, 50, 10, weight_init, optimizer);

    let mut losses = vec![];

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

        let _loss = model.loss(&x_batch, &t_batch);
        model.backward();

        let loss = model.loss(x_test, t_test);
        // println!("elapsed: 1_1= {elapsed11} 1= {elapsed1}, 2= {elapsed2}, 3= {elapsed3}");

        // println!("idx: {i} loss: {loss}");

        losses.push(loss);
    }

    losses
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

    let iters_num = 5000;
    let train_size = x_train.shape()[0];
    let batch_size = 100;

    let iter_per_epoch = std::cmp::max(train_size / batch_size as usize, 1);

    let mut rng = rand::rng();

    let mut losses_map = BTreeMap::new();

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
        (WeightInit::He, Optimizer::Sgd(Sgd::new(0.01))),
        (
            WeightInit::He,
            Optimizer::Momentum(Momentum::new(0.01, 0.9)),
        ),
        (WeightInit::He, Optimizer::AdaGrad(AdaGrad::new(0.01))),
        (
            WeightInit::He,
            Optimizer::Adam(Adam::new(0.001, 0.9, 0.999)),
        ),
    ];

    for (weight_init, optimizer) in comb {
        let begin = Instant::now();

        let losses = model_train(
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
        );

        let elapsed = begin.elapsed().as_secs_f32();
        println!("elapsed: {elapsed} weight_init: {weight_init:?} optimizer: {optimizer:?}");

        // let mut maps = BTreeMap::new();
        // maps.insert(format!("{optimizer:?}"), losses);

        losses_map
            .entry(format!("{weight_init:?}"))
            .or_insert_with(|| BTreeMap::default())
            .insert(format!("{optimizer:?}"), losses);
    }

    let app = LossApp { losses_map };

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
    losses_map: BTreeMap<String, BTreeMap<String, Vec<f32>>>,
}

impl eframe::App for LossApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let legend = Legend::default().position(egui_plot::Corner::RightTop);
            egui_plot::Plot::new(format!("plot"))
                .legend(legend)
                .show(ui, |plot_ui| {
                    for (optim_name, weight_losses) in &self.losses_map {
                        for (weight_generator, losses) in weight_losses {
                            let points = losses
                                .into_iter()
                                .enumerate()
                                .map(|(idx, loss)| [idx as f64, *loss as f64]);

                            let points = PlotPoints::from_iter(points);

                            plot_ui.points(
                                Points::new("optim", points)
                                    .name(format!("{optim_name}_{weight_generator}")),
                            );
                        }
                    }
                });
        });
    }
}
