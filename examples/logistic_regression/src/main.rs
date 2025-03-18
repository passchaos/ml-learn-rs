extern crate openblas_src;

use std::io::BufRead;

use egui_plot::{Legend, PlotPoints, Points};
use ndarray::Array2;

fn load_data_set() -> (Array2<f64>, Array2<f64>) {
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
        Array2::from_shape_vec((m, 1), label_vec).unwrap(),
    )
}

fn main() {
    let (data_in, labels_in) = load_data_set();

    let weights = alg::logistic::gradient_ascent(data_in.view(), labels_in.view());

    let mut data = vec![];
    let mut labels = vec![];

    for (i_data, i_label) in data_in
        .axis_iter(ndarray::Axis(0))
        .zip(labels_in.axis_iter(ndarray::Axis(0)))
    {
        data.push([i_data[1], i_data[2]]);
        labels.push(i_label[0]);
    }

    println!("weights: {:?}", weights);

    let weights = weights.index_axis(ndarray::Axis(1), 0).to_vec();

    plot_data(data, labels, weights);
}

fn plot_data(data: Vec<[f64; 2]>, labels: Vec<f64>, weights: Vec<f64>) {
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

struct App {
    weights: Vec<f64>,
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
                    let line_data: Vec<_> = ndarray::linspace(-3.0, 3.0, 100)
                        .map(|x| {
                            let y = -(self.weights[0] + self.weights[1] * x) / self.weights[2];

                            [x, y]
                        })
                        .collect();

                    let line = egui_plot::Line::new(PlotPoints::from(line_data))
                        .color(egui::Color32::GREEN);
                    plot_ui.line(line);

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
