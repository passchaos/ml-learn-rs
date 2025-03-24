extern crate openblas_src;

use std::{collections::HashMap, io::BufRead};

use alg::math::Sigmoid;
use anyhow::Result;
use egui::vec2;
use egui_plot::{Legend, PlotPoints, Points};
use ndarray::{Array1, Array2, ArrayView1, Axis, arr0, arr1};

fn load_data_set() -> (Array2<f64>, Array1<f64>) {
    let path = tools::full_file_path("Ch05/testSet.txt");

    let file = std::io::BufReader::new(std::fs::File::open(path).unwrap());

    let mut data_mat = vec![];
    let mut label_vec = vec![];

    let mut m = 0;
    for line in file.lines() {
        let line = line.unwrap();

        let mut a = line.trim().split_whitespace();

        // println!("a: {a:?}");

        let line_0 = a.next().unwrap().parse().unwrap();
        let line_1 = a.next().unwrap().parse().unwrap();
        let line_2: f64 = a.next().unwrap().parse().unwrap();

        data_mat.extend([1.0, line_0, line_1]);
        label_vec.push(line_2);
        m += 1;
    }

    (
        Array2::from_shape_vec((m, 3), data_mat).unwrap(),
        Array1::from_vec(label_vec),
    )
}

fn main() {
    // gradient_descent_plot();

    let num_tests = 10;
    let mut err_sum = 0.0;

    for _ in 0..num_tests {
        let err_rate = colic_test().unwrap();
        err_sum += err_rate;
    }

    println!(
        "after {num_tests} iterations, the average error rate is: {}",
        err_sum / num_tests as f64
    );
}

fn gradient_descent_plot() {
    let (data_in, labels_in) = load_data_set();

    // let mut weights_iterations: Vec<Array1<f64>> = vec![];

    // let weights = alg::logistic::gradient_ascent(data_in.view(), labels_in.view());
    // let weights = alg::logistic::stoc_grad_ascent_0(
    //     &mut weights_iterations,
    //     data_in.view(),
    //     labels_in.view(),
    //     200,
    // );

    // let weights = alg::logistic::stoc_grad_ascent_1(
    //     &mut weights_iterations,
    //     data_in.view(),
    //     labels_in.view(),
    //     1000,
    // );
    // println!("weights: {weights:?}");

    // plot_weights_iterations(weights_iterations);

    let mut weights_map = HashMap::new();
    for count in [200, 500, 1000] {
        let weights = alg::logistic::gradient_ascent(data_in.view(), labels_in.view(), count);
        weights_map.insert(
            (
                format!("Gradient ascent: {count}"),
                egui::Color32::DARK_GREEN,
            ),
            weights,
        );

        let (weights, _) =
            alg::logistic::stoc_grad_ascent_1(data_in.view(), labels_in.view(), count);

        weights_map.insert(
            (
                format!("RandomGradicent ascent: {count}"),
                egui::Color32::LIGHT_GREEN,
            ),
            weights,
        );
    }

    plot_data(data_in, labels_in, weights_map);
}

fn plot_weights_iterations(weights_iterations: Vec<Array1<f64>>) {
    let weights_iterations = weights_iterations
        .into_iter()
        .map(|r| vec![r[0], r[1], r[2]])
        .collect();

    let app = IterationState { weights_iterations };

    eframe::run_native(
        "Plot",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Ok(Box::new(app))),
    )
    .unwrap();
}

fn plot_data(
    data_in: Array2<f64>,
    labels_in: Array1<f64>,
    weights_map: HashMap<(String, egui::Color32), Array1<f64>>,
) {
    let mut data = vec![];
    let mut labels = vec![];

    for (i_data, i_label) in data_in
        .axis_iter(ndarray::Axis(0))
        .zip(labels_in.into_iter())
    {
        data.push([i_data[1], i_data[2]]);
        labels.push(i_label);
    }

    let weights = weights_map
        .into_iter()
        .map(|(k, v)| (k, v.to_vec()))
        .collect();

    let app = App {
        weights,
        data,
        labels,
    };

    eframe::run_native(
        "Plot",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Ok(Box::new(app))),
    )
    .unwrap();
}

struct IterationState {
    weights_iterations: Vec<Vec<f64>>,
}

impl eframe::App for IterationState {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let mut x0_iteration = vec![];
            let mut x1_iteration = vec![];
            let mut x2_iteration = vec![];

            for (idx, weight) in self.weights_iterations.iter().enumerate() {
                x0_iteration.push([idx as f64, weight[0]]);
                x1_iteration.push([idx as f64, weight[1]]);
                x2_iteration.push([idx as f64, weight[2]]);
            }

            let rect1 = ui.max_rect();

            let height = rect1.height() / 3.0;
            let offset = vec2(rect1.width(), height);

            ui.allocate_ui(offset, |ui| {
                egui_plot::Plot::new("x0 plot")
                    .y_axis_label("X0")
                    .show(ui, |plot_ui| {
                        let points = PlotPoints::new(x0_iteration);

                        plot_ui.points(Points::new(points));
                    });
            });

            ui.allocate_ui(offset, |ui| {
                egui_plot::Plot::new("x1 plot")
                    .y_axis_label("X1")
                    .show(ui, |plot_ui| {
                        let points = PlotPoints::new(x1_iteration);

                        plot_ui.points(Points::new(points));
                    });
            });

            ui.allocate_ui(offset, |ui| {
                egui_plot::Plot::new("x2 plot")
                    .y_axis_label("X2")
                    .show(ui, |plot_ui| {
                        let points = PlotPoints::new(x2_iteration);

                        plot_ui.points(Points::new(points));
                    })
            });
        });
    }
}

struct App {
    weights: HashMap<(String, egui::Color32), Vec<f64>>,
    data: Vec<[f64; 2]>,
    labels: Vec<f64>,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let legend = Legend::default().position(egui_plot::Corner::RightTop);

            let data = self
                .data
                .clone()
                .into_iter()
                .zip(self.labels.clone().into_iter());

            egui_plot::Plot::new("plot")
                .legend(legend)
                .show(ui, |plot_ui| {
                    for ((label, color), weights) in &self.weights {
                        let line_data: Vec<_> = ndarray::linspace(-3.0, 3.0, 100)
                            .map(|x| {
                                let y = -(weights[0] + weights[1] * x) / weights[2];

                                [x, y]
                            })
                            .collect();

                        let line = egui_plot::Line::new(PlotPoints::from(line_data))
                            .name(format!("{label}"))
                            .color(color.clone());
                        plot_ui.line(line);
                    }

                    for (point, label) in data {
                        let points = PlotPoints::from(point);

                        let radius = if label == 1.0 { 2.0 } else { 5.0 };

                        plot_ui.points(
                            Points::new(points)
                                .radius(radius)
                                .color(egui::Color32::LIGHT_RED),
                        );
                    }
                });
        });
    }
}

fn classify_vector(inx: ArrayView1<f64>, weights: &Array1<f64>) -> f64 {
    let prob = inx.dot(weights).sigmoid();

    if prob > 0.5 { 1.0 } else { 0.0 }
}

fn colic_test() -> Result<f64> {
    let train_file = std::io::BufReader::new(std::fs::File::open(tools::full_file_path(
        "Ch05/horseColicTraining.txt",
    ))?);
    let test_file = std::io::BufReader::new(std::fs::File::open(tools::full_file_path(
        "Ch05/horseColicTest.txt",
    ))?);

    const ATT_COUNT: usize = 21;

    let mut train_set = Array2::zeros((0, ATT_COUNT));
    let mut train_labels = Array1::from_vec(vec![]);

    for line_str in train_file.lines() {
        let line_s = line_str.unwrap();

        let mut curr_line = line_s.trim().split('\t');

        let mut line_att = vec![];

        for _ in 0..ATT_COUNT {
            let f = curr_line.next().unwrap().parse::<f64>().unwrap();
            line_att.push(f);
        }

        let new_line_att = arr1(&line_att);

        train_set.push(Axis(0), new_line_att.view()).unwrap();

        let label = curr_line.next().unwrap().parse::<f64>().unwrap();
        train_labels.push(Axis(0), arr0(label).view()).unwrap();
    }

    let (train_weights, _) =
        alg::logistic::stoc_grad_ascent_1(train_set.view(), train_labels.view(), 500);
    println!("train weights: {train_weights}");

    let mut error_count = 0.0;
    let mut num_test_vec = 0.0;

    for line in test_file.lines() {
        num_test_vec += 1.0;

        let line_s = line.unwrap();
        let mut curr_line = line_s.trim().split('\t');

        let mut line_att = vec![];
        for _ in 0..ATT_COUNT {
            let f = curr_line.next().unwrap().parse::<f64>().unwrap();
            line_att.push(f);
        }

        let label = curr_line.next().unwrap().parse::<f64>().unwrap();

        let inx = Array1::from_vec(line_att);
        let classified_label = classify_vector(inx.view(), &train_weights);
        // println!("label: {label} classified_label: {classified_label}");

        if label != classified_label {
            error_count += 1.0;
        }
    }

    let err_rate = error_count / num_test_vec;
    println!("error rate: {err_rate}");

    Ok(err_rate)
}
