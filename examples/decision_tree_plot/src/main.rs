use anyhow::Result;
use egui::Image;

fn main() {
    eframe::run_native(
        "Decision Tree Plot",
        eframe::NativeOptions::default(),
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            tools::add_font(&cc.egui_ctx);
            Ok(Box::new(DecisionTreePlot::default()))
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
    fn new(_cc: &eframe::CreationContext) -> Self {
        Self::default()
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
                println!("svg_s len: {:?}", dts.as_ref().map(|s| s.len()));

                match dts {
                    Ok(svg_d) => {
                        let s = tools::md5_hex(svg_d);

                        let image = Image::from_bytes(
                            format!("bytes://abc_{s}.svg",),
                            svg_d.to_string().into_bytes(),
                        );

                        ui.add_sized([800.0, 1000.0], image);
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
