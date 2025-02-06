extern crate ml_learn_rs;

use egui::Color32;
use egui_plot::{Legend, PlotPoints, Points};
use ml_learn_rs::{knn, tools};
use ndarray::{Array1, ArrayView1, array, s};

fn main() {
    tools::init_logs();

    let input = array![0.0, 0.0];
    let data_set = array![[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]];
    let labels = array![
        "A".to_string(),
        "A".to_string(),
        "B".to_string(),
        "B".to_string()
    ];

    let res = knn::classify(input.view(), data_set.view(), labels.view(), 3);
    tracing::info!("result: {res}");

    let (data, labels) = knn::file2matrix(tools::full_file_path("Ch02/datingTestSet2.txt"));

    let (normed_data, ranges, min_vals) = knn::auto_norm(data.view());

    let a = normed_data.slice(s![.., 0]);
    let b = normed_data.slice(s![.., 1]);
    let c = normed_data.slice(s![.., 2]);

    let weight: Array1<u8> = labels.iter().map(|a| a.parse().unwrap()).collect();

    let data = PlotDemo {
        a: (a, "每年获得的飞行客里程数"),
        b: (b, "玩视频游戏所耗时间百分比"),
        c: (c, "每周消费的冰激凌公升数"),
        weight,
        relation: Relation::AB,
    };

    println!(
        "fonts info: {:?}",
        egui::FontDefinitions::default().families
    );

    eframe::run_native(
        "Plot",
        eframe::NativeOptions::default(),
        Box::new(|cc| {
            add_font(&cc.egui_ctx);
            Ok(Box::new(data))
        }),
    )
    .unwrap();
}

#[derive(PartialEq)]
enum Relation {
    AB,
    AC,
    BC,
}

struct PlotDemo<'a> {
    a: (ArrayView1<'a, f64>, &'a str),
    b: (ArrayView1<'a, f64>, &'a str),
    c: (ArrayView1<'a, f64>, &'a str),
    weight: Array1<u8>,
    relation: Relation,
}

impl<'a> eframe::App for PlotDemo<'a> {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("side_panel").show(ctx, |ui| {
            ui.heading("炼丹");

            ui.vertical(|ui| {
                ui.radio_value(&mut self.relation, Relation::AB, "A_B");
                ui.radio_value(&mut self.relation, Relation::AC, "A_C");
                ui.radio_value(&mut self.relation, Relation::BC, "B_C");
            });
        });

        let (x, y) = match self.relation {
            Relation::AB => (self.a, self.b),
            Relation::AC => (self.a, self.c),
            Relation::BC => (self.b, self.c),
        };

        let data =
            x.0.iter()
                .zip(y.0.iter())
                .map(|(a, b)| [*a, *b])
                .zip(self.weight.iter());

        egui::CentralPanel::default().show(ctx, |ui| {
            let legend = Legend::default().position(egui_plot::Corner::RightTop);

            egui_plot::Plot::new("plot")
                .legend(legend)
                .x_axis_label(x.1)
                .y_axis_label(y.1)
                .show(ui, |plot_ui| {
                    for (point, weight) in data {
                        let sine_points = PlotPoints::new(vec![point]);

                        let radius = 4.0 * *weight as f32;

                        let r = 75;
                        let g = 38;
                        let b = 46;

                        let color = Color32::from_rgb(r * weight, g * weight, b * weight);

                        let label = match *weight {
                            1 => "不喜欢",
                            2 => "魅力一般",
                            _ => "极具魅力",
                        };

                        plot_ui.points(
                            Points::new(sine_points)
                                .name(label)
                                .radius(radius)
                                .color(color),
                        );
                    }
                });
        });
    }
}

fn add_font(ctx: &egui::Context) {
    let mut font_definitions = egui::FontDefinitions::default();

    let font_data = egui::FontData::from_static(include_bytes!("../assets/fonts/PingFang.ttc"));

    font_definitions
        .font_data
        .insert("pingfang".to_string(), std::sync::Arc::new(font_data));

    font_definitions
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "pingfang".to_string());

    ctx.set_fonts(font_definitions);
}
