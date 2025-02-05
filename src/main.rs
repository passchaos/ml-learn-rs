use anyhow::Result;
use egui::Color32;
use egui_plot::{PlotPoints, Points};
use ndarray::{Array, Array1, ArrayBase, ArrayView1, ViewRepr, arr2, array, s};
use plotly::{
    Layout, Plot, Scatter,
    common::{Marker, Mode, Title},
    layout::{Axis, Legend},
};

mod knn;

fn main() -> Result<()> {
    let input = array![0.0, 0.0];
    let data_set = array![[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]];
    let labels = array![
        "A".to_string(),
        "A".to_string(),
        "B".to_string(),
        "B".to_string()
    ];

    let res = knn::classify(input, data_set, labels, 3);
    println!("result: {res}");

    let file_path = format!(
        "{}/Work/ml-learn-rs/MachineLearningInActionSourceCode/Ch02/datingTestSet2.txt",
        dirs::home_dir().unwrap().to_str().unwrap()
    );

    let (data, labels) = knn::file2matrix(&file_path);

    let a = data.slice(s![.., 0]);
    let (a, a_range, a_min) = knn::auto_norm(a);

    let b = data.slice(s![.., 1]);
    let (b, b_range, b_min) = knn::auto_norm(b);

    let c = data.slice(s![.., 2]);
    let (c, c_range, c_min) = knn::auto_norm(c);

    let weight: Array1<u8> = labels.iter().map(|a| a.parse().unwrap()).collect();

    let data = PlotDemo {
        a: (a.view(), "每年获得的飞行客里程数"),
        b: (b.view(), "玩视频游戏所耗时间百分比"),
        c: (c.view(), "每周消费的冰激凌公升数"),
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

    Ok(())
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
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::SidePanel::left("side_panel").show(ctx, |ui| {
            ui.heading("Settings");

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
            egui_plot::Plot::new("plot")
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

                        plot_ui.points(
                            Points::new(sine_points)
                                .name("Sine")
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

    font_definitions.font_data.insert("pingfang".to_string(),
        std::sync::Arc::new(egui::FontData::from_static(include_bytes!("/System/Library/AssetsV2/com_apple_MobileAsset_Font7/3419f2a427639ad8c8e139149a287865a90fa17e.asset/AssetData/PingFang.ttc"))));

    font_definitions
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "pingfang".to_string());

    ctx.set_fonts(font_definitions);
}
