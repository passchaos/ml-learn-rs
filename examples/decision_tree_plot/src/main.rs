use std::sync::Arc;

use anyhow::Result;
use egui::{Image, Sense, TextWrapMode};
use egui_snarl::{
    Snarl,
    ui::{NodeLayout, PinPlacement, SnarlStyle, SnarlViewer},
};

fn dot_to_svg(dot_s: &str) -> Result<String> {
    let mut parser = layout::gv::DotParser::new(&dot_s);

    let g = parser.process().map_err(|e| anyhow::anyhow!(e))?;

    let mut g_b = layout::gv::GraphBuilder::new();
    g_b.visit_graph(&g);
    let mut vg = g_b.get();

    let mut sw = layout::backends::svg::SVGWriter::new();
    vg.do_it(false, false, false, &mut sw);

    Ok(sw.finalize())
}

fn main() {
    let svg_s = dot_to_svg(
        r#"digraph G {
            a -> b [label = "天涯"];
            }"#,
    )
    .unwrap();

    std::fs::write("/tmp/output.svg", &svg_s).unwrap();

    eframe::run_native(
        "Decision Tree Plot",
        eframe::NativeOptions::default(),
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Ok(Box::new(DecisionTreePlot::default()))
        }),
    )
    .unwrap();
    println!("Hello, world!");
}

#[derive(Debug)]
enum Node {
    A,
    B,
    C,
}

#[derive(Debug, Default)]
struct DecisionTreeViewer;

impl SnarlViewer<Node> for DecisionTreeViewer {
    fn title(&mut self, node: &Node) -> String {
        format!("node_{node:?}")
    }

    fn inputs(&mut self, node: &Node) -> usize {
        1
    }

    fn show_input(
        &mut self,
        pin: &egui_snarl::InPin,
        ui: &mut egui::Ui,
        scale: f32,
        snarl: &mut Snarl<Node>,
    ) -> egui_snarl::ui::PinInfo {
        ui.label("parent");
        egui_snarl::ui::PinInfo::default()
    }

    fn outputs(&mut self, node: &Node) -> usize {
        1
    }

    fn show_output(
        &mut self,
        pin: &egui_snarl::OutPin,
        ui: &mut egui::Ui,
        scale: f32,
        snarl: &mut Snarl<Node>,
    ) -> egui_snarl::ui::PinInfo {
        ui.label("child");
        egui_snarl::ui::PinInfo::default()
    }

    fn has_dropped_wire_menu(
        &mut self,
        src_pins: egui_snarl::ui::AnyPins,
        snarl: &mut Snarl<Node>,
    ) -> bool {
        true
    }

    fn show_dropped_wire_menu(
        &mut self,
        pos: egui::Pos2,
        ui: &mut egui::Ui,
        scale: f32,
        src_pins: egui_snarl::ui::AnyPins,
        snarl: &mut Snarl<Node>,
    ) {
        ui.label("wire context");
    }

    fn has_wire_widget(
        &mut self,
        from: &egui_snarl::OutPinId,
        to: &egui_snarl::InPinId,
        snarl: &Snarl<Node>,
    ) -> bool {
        true
    }

    fn show_wire_widget(
        &mut self,
        from: &egui_snarl::OutPin,
        to: &egui_snarl::InPin,
        ui: &mut egui::Ui,
        scale: f32,
        snarl: &mut Snarl<Node>,
    ) {
        ui.label("annotation");
    }

    fn has_body(&mut self, node: &Node) -> bool {
        true
    }

    fn show_body(
        &mut self,
        node: egui_snarl::NodeId,
        inputs: &[egui_snarl::InPin],
        outputs: &[egui_snarl::OutPin],
        ui: &mut egui::Ui,
        scale: f32,
        snarl: &mut Snarl<Node>,
    ) {
        ui.label("body container");
    }

    fn has_graph_menu(&mut self, pos: egui::Pos2, snarl: &mut Snarl<Node>) -> bool {
        true
    }

    fn show_graph_menu(
        &mut self,
        pos: egui::Pos2,
        ui: &mut egui::Ui,
        scale: f32,
        snarl: &mut Snarl<Node>,
    ) {
        ui.label("Menu node");

        if ui.button("Add node A").clicked() {
            snarl.insert_node(pos, Node::A);
            ui.close_menu();
        }

        if ui.button("Add node B").clicked() {
            snarl.insert_node(pos, Node::B);
            ui.close_menu();
        }
    }
}

#[derive(Debug, Default, PartialEq)]
enum Enum {
    #[default]
    First,
    Second,
    Third,
}

#[derive(Debug, Default)]
struct DecisionTreePlot {
    snarl: Snarl<Node>, // Define fields here
    viewer: DecisionTreeViewer,
    dot_content: String,
    svg_s: Option<Result<String>>,
    counter: i32,
    text: String,
    my_f32: f32,
    my_boolean: bool,
    my_enum: Enum,
    allowed_to_close: bool,
    show_confirmation_dialog: bool,
}

impl DecisionTreePlot {
    fn new(cc: &eframe::CreationContext) -> Self {
        tools::add_font(&cc.egui_ctx);
        // Self {
        //     svg: svg.into_bytes(),
        //     ..Default::default()
        // }
        Self::default()
        // let snarl = Snarl::new();
        // let viewer = DecisionTreeViewer;
        // // Initialize fields here
        // DecisionTreePlot {
        //     snarl,
        //     viewer,
        //     counter: 10,
        //     text: "default text".to_string(),
        // }
    }
}

impl eframe::App for DecisionTreePlot {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // ui.heading("Decision Tree Plot");

            // ui.horizontal(|ui| {
            //     if ui.button("-").clicked() {
            //         self.counter -= 1;
            //     }
            //     ui.label(self.counter.to_string());
            //     if ui.button("+").clicked() {
            //         self.counter += 1;
            //     }
            // });

            // ui.hyperlink("https://baidu.com");
            // ui.text_edit_singleline(&mut self.text);
            // // ui.add(egui::Slider::new(&mut self.my_f32, 0.0..=100.0));
            // //
            // ui.add_sized([240.0, 120.0], egui::DragValue::new(&mut self.my_f32));
            // ui.add(egui::DragValue::new(&mut self.my_f32));

            // ui.horizontal(|ui| {
            //     ui.radio_value(&mut self.my_enum, Enum::First, "First value");
            //     ui.radio_value(&mut self.my_enum, Enum::Second, "Second value");
            //     ui.radio_value(&mut self.my_enum, Enum::Third, "Third value");
            //     ui.radio_value(&mut self.my_enum, Enum::First, "First value");
            //     ui.radio_value(&mut self.my_enum, Enum::Second, "Second value");
            //     ui.radio_value(&mut self.my_enum, Enum::Third, "Third value");
            //     ui.radio_value(&mut self.my_enum, Enum::First, "First value");
            //     // ui.radio_value(&mut self.my_enum, Enum::Second, "Second value");
            //     // ui.radio_value(&mut self.my_enum, Enum::Third, "Third value");
            //     // ui.radio_value(&mut self.my_enum, Enum::First, "First value");
            //     // ui.radio_value(&mut self.my_enum, Enum::Second, "Second value");
            //     // ui.radio_value(&mut self.my_enum, Enum::Third, "Third value");
            // });

            // ui.horizontal_wrapped(|ui| {
            //     ui.spacing_mut().item_spacing.x = 0.0;
            //     ui.radio_value(&mut self.my_enum, Enum::First, "First value");
            //     ui.radio_value(&mut self.my_enum, Enum::Second, "Second value");
            //     ui.radio_value(&mut self.my_enum, Enum::Third, "Third value");
            //     ui.radio_value(&mut self.my_enum, Enum::First, "First value");
            //     ui.radio_value(&mut self.my_enum, Enum::Second, "Second value");
            //     ui.radio_value(&mut self.my_enum, Enum::Third, "Third value");
            //     ui.radio_value(&mut self.my_enum, Enum::First, "First value");
            //     ui.radio_value(&mut self.my_enum, Enum::Second, "Second value");
            //     ui.radio_value(&mut self.my_enum, Enum::Third, "Third value");
            //     ui.radio_value(&mut self.my_enum, Enum::First, "First value");
            //     ui.radio_value(&mut self.my_enum, Enum::Second, "Second value");
            //     ui.radio_value(&mut self.my_enum, Enum::Third, "Third value");
            // });

            // //
            // ui.separator();

            // ui.collapsing("Click to see what is hidden!", |ui| {
            //     if ui.button("haha").clicked() {
            //         ui.label("haha");
            //     }
            //     ui.label("Not much, as it turns out");
            // });

            // ui.group(|ui| {
            //     ui.label("Within a frame");
            //     ui.set_min_height(200.0);
            //     if ui.button("help").clicked() {
            //         ui.label("ddd");
            //     }
            // });

            // ui.scope(|ui| {
            //     ui.visuals_mut().override_text_color = Some(egui::Color32::RED);
            //     ui.style_mut().override_text_style = Some(egui::TextStyle::Monospace);
            //     ui.style_mut().wrap_mode = Some(TextWrapMode::Truncate);

            //     ui.label("This text will be red, monospace, and won't wrap to a new line");
            // });
            //

            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label("graphviz content:");
                    ui.text_edit_multiline(&mut self.dot_content);
                    if ui.button("preview").clicked() {
                        let dts = dot_to_svg(&self.dot_content);

                        self.svg_s = Some(dts);
                    }
                });

                if let Some(dts) = self.svg_s.as_ref() {
                    println!("svg_s len: {:?}", dts.as_ref().map(|s| s.len()));

                    match dts {
                        Ok(svg_d) => {
                            use md5::Digest;
                            let mut hasher = md5::Md5::new();
                            hasher.update(svg_d.as_bytes());
                            let s = hex::encode(hasher.finalize());

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
            // self.snarl
            //     .show(&mut self.viewer, &default_style(), "decision_tree", ui);
        });

        if ctx.input(|i| i.viewport().close_requested()) {
            if self.allowed_to_close {
            } else {
                ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                self.show_confirmation_dialog = true;
            }
        }

        if self.show_confirmation_dialog {
            egui::Window::new("Do you want to quit?")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        if ui.button("No").clicked() {
                            self.show_confirmation_dialog = false;
                            self.allowed_to_close = false;
                        }

                        if ui.button("Yes").clicked() {
                            self.show_confirmation_dialog = false;
                            self.allowed_to_close = true;
                            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    })
                });
        }
    }
}

const fn default_style() -> SnarlStyle {
    SnarlStyle {
        node_layout: Some(NodeLayout::Basic),
        pin_placement: Some(PinPlacement::Edge),
        pin_size: Some(7.0),
        node_frame: Some(egui::Frame {
            inner_margin: egui::Margin::same(8.0),
            outer_margin: egui::Margin {
                left: 0.0,
                right: 0.0,
                top: 0.0,
                bottom: 4.0,
            },
            rounding: egui::Rounding::same(8.0),
            fill: egui::Color32::from_gray(30),
            stroke: egui::Stroke::NONE,
            shadow: egui::Shadow::NONE,
        }),
        bg_frame: Some(egui::Frame {
            inner_margin: egui::Margin::same(2.0),
            outer_margin: egui::Margin::ZERO,
            rounding: egui::Rounding::ZERO,
            fill: egui::Color32::from_gray(40),
            stroke: egui::Stroke::NONE,
            shadow: egui::Shadow::NONE,
        }),
        ..SnarlStyle::new()
    }
}
