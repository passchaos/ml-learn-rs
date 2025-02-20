use std::{
    collections::HashMap,
    io::{BufRead, BufReader},
};

use ndarray::{Array2, array};

use anyhow::Result;
use egui::Image;

fn main() {
    tools::init_log();

    let file = tools::full_file_path("Ch03/lenses.txt");

    let reader = BufReader::new(std::fs::File::open(file).unwrap());

    let mut nrows = 0;

    let data = reader
        .lines()
        .into_iter()
        .map(|line| {
            nrows += 1;
            let line = line.unwrap().trim().to_string();
            line.split('\t').map(|a| a.to_string()).collect::<Vec<_>>()
        })
        .flatten()
        .collect::<Vec<_>>();

    let data_set = Array2::from_shape_vec((nrows, 5), data).unwrap();
    let features = array!["age", "prescript", "astigmatic", "tearRate"].map(|a| a.to_string());

    let mut map = HashMap::new();

    alg::decision_tree::create_tree(data_set.view(), &mut features.clone(), &mut map);
    tracing::info!("map: {map:#?}");

    let dot_c = alg::decision_tree::tree_to_dot_content(&map);
    tracing::info!("dot: {dot_c}");

    eframe::run_native(
        "Decision Tree Plot",
        eframe::NativeOptions::default(),
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            tools::add_font(&cc.egui_ctx);
            Ok(Box::new(DecisionTreePlot::new(cc, dot_c)))
        }),
    )
    .unwrap();
}

#[derive(Debug, Default)]
struct DecisionTreePlot {
    dot_content: String,
    svg_s: Option<Result<String>>,
}

impl DecisionTreePlot {
    fn new(_cc: &eframe::CreationContext, dot_content: String) -> Self {
        Self {
            dot_content,
            ..Default::default()
        }
    }

    fn plot_graphviz(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label("graphviz content:");

                let text_edit = egui::TextEdit::multiline(&mut self.dot_content)
                    .code_editor()
                    .background_color(egui::Color32::LIGHT_GRAY);
                ui.add(text_edit);
                // ui.code_editor(&mut self.dot_content);

                // ui.text_edit_multiline(&mut self.dot_content);
                if ui.button("preview").clicked() {
                    let dts = tools::dot_to_svg(&self.dot_content);

                    self.svg_s = Some(dts);
                }
            });

            if let Some(dts) = self.svg_s.as_ref() {
                match dts {
                    Ok(svg_d) => {
                        let s = tools::md5_hex(svg_d);

                        let image = Image::from_bytes(
                            format!("bytes://abc_{s}.svg",),
                            svg_d.to_string().into_bytes(),
                        );

                        ui.add_sized([1000.0, 1000.0], image);
                        // ui.add(image);
                    }
                    Err(e) => {
                        ui.colored_label(
                            egui::Color32::RED,
                            format!("parse dot meet failure: err= {e}"),
                        );
                    }
                }
            }
        });
    }
}

impl eframe::App for DecisionTreePlot {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let frame = egui::containers::Frame {
            fill: egui::Color32::WHITE,
            ..Default::default()
        };
        egui::CentralPanel::default().frame(frame).show(ctx, |ui| {
            self.plot_graphviz(ui);
        });
    }
}
