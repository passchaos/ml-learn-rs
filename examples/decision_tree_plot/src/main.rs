use std::collections::HashMap;

use alg::decision_tree::MapValue;
use ndarray::array;

use anyhow::Result;
use egui::Image;

fn main() {
    tracing_subscriber::fmt().init();

    let data_set = array![
        ["1", "1", "yes"],
        ["1", "1", "yes"],
        ["1", "0", "no"],
        ["0", "1", "no"],
        ["0", "1", "no"],
    ]
    .map(|a| a.to_string());

    let mut map = HashMap::new();

    let mut features = array!["no surfacing", "flippers"].mapv(|a| a.to_string());

    alg::decision_tree::create_tree(data_set.view(), &mut features, &mut map);
    tracing::info!("map: {map:#?}");

    let mut inner_map = HashMap::new();
    inner_map.insert("maybe".to_string(), MapValue::default());

    map.get_mut("no surfacing")
        .unwrap()
        .map
        .insert("3".to_string(), MapValue::from(inner_map));

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

                let text_edit = egui::TextEdit::multiline(&mut self.dot_content).code_editor();
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

                        ui.add_sized([600.0, 600.0], image);
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
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            self.plot_graphviz(ui);
        });
    }
}
