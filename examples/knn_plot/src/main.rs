use bevy::prelude::*;
use bevy::{
    DefaultPlugins,
    app::{App, Update},
};
use bevy_egui::{EguiContexts, EguiPlugin};
use egui::Color32;
use egui_plot::{Legend, PlotPoints, Points};
use ndarray::{Array1, Array2, array, s};

use alg::knn;

fn main() {
    tracing_subscriber::fmt().init();

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

    let (normed_data, _ranges, _min_vals) = knn::auto_norm(data.view());

    let weight: Array1<u8> = labels.iter().map(|a| a.parse().unwrap()).collect();

    let data = UiState {
        data: normed_data,
        // a: (a, "每年获得的飞行客里程数"),
        // b: (b, "玩视频游戏所耗时间百分比"),
        // c: (c, "每周消费的冰激凌公升数"),
        weight,
        relation: Relation::AB,
    };

    println!(
        "fonts info: {:?}",
        egui::FontDefinitions::default().families
    );

    App::new()
        .insert_resource(data)
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin)
        .add_systems(Update, ui_example_system)
        .add_systems(Startup, setup)
        .run();

    // eframe::run_native(
    //     "Plot",
    //     eframe::NativeOptions::default(),
    //     Box::new(|cc| {
    //         add_font(&cc.egui_ctx);
    //         Ok(Box::new(data))
    //     }),
    // )
    // .unwrap();
}

fn setup(mut contexts: EguiContexts) {
    let mut visuals = egui::Visuals::light();
    visuals.faint_bg_color = Color32::from_rgb(255, 255, 255);
    contexts.ctx_mut().set_style(egui::Style {
        visuals,
        ..Default::default()
    });

    tools::add_font(contexts.ctx_mut());
}

fn ui_example_system(mut ui_state: ResMut<UiState>, mut contexts: EguiContexts) {
    ui_state.as_mut().plot(contexts.ctx_mut());
}

#[derive(PartialEq)]
enum Relation {
    AB,
    AC,
    BC,
}

#[derive(Resource)]
struct UiState {
    data: Array2<f64>,
    weight: Array1<u8>,
    relation: Relation,
}

impl eframe::App for UiState {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.plot(ctx);
    }
}

impl UiState {
    fn plot(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("side_panel").show(ctx, |ui| {
            ui.heading("炼丹");

            ui.vertical(|ui| {
                ui.radio_value(&mut self.relation, Relation::AB, "A_B");
                ui.radio_value(&mut self.relation, Relation::AC, "A_C");
                ui.radio_value(&mut self.relation, Relation::BC, "B_C");
            });
        });

        let a = self.data.slice(s![.., 0]);
        let b = self.data.slice(s![.., 1]);
        let c = self.data.slice(s![.., 2]);

        let a = (a, "每年获得的飞行客里程数");
        let b = (b, "玩视频游戏所耗时间百分比");
        let c = (c, "每周消费的冰激凌公升数");

        let (x, y) = match self.relation {
            Relation::AB => (a, b),
            Relation::AC => (a, c),
            Relation::BC => (b, c),
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
