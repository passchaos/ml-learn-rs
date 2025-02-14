use egui_snarl::{
    Snarl,
    ui::{NodeLayout, PinPlacement, SnarlStyle, SnarlViewer},
};

fn main() {
    eframe::run_native(
        "Decision Tree Plot",
        eframe::NativeOptions::default(),
        Box::new(|cc| Ok(Box::new(DecisionTreePlot::new(cc)))),
    )
    .unwrap();
    println!("Hello, world!");
}

enum Node {
    A,
    B,
}

struct DecisionTreeViewer;

impl SnarlViewer<Node> for DecisionTreeViewer {
    fn title(&mut self, node: &Node) -> String {
        "decision_tree".to_string()
    }

    fn inputs(&mut self, node: &Node) -> usize {
        0
    }

    fn show_input(
        &mut self,
        pin: &egui_snarl::InPin,
        ui: &mut egui::Ui,
        scale: f32,
        snarl: &mut Snarl<Node>,
    ) -> egui_snarl::ui::PinInfo {
        todo!()
    }

    fn outputs(&mut self, node: &Node) -> usize {
        0
    }

    fn show_output(
        &mut self,
        pin: &egui_snarl::OutPin,
        ui: &mut egui::Ui,
        scale: f32,
        snarl: &mut Snarl<Node>,
    ) -> egui_snarl::ui::PinInfo {
        todo!()
    }

    fn has_body(&mut self, node: &Node) -> bool {
        true
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

        if ui.button("Add node").clicked() {
            snarl.insert_node(pos, Node::A);
            ui.close_menu();
        }
    }
}

struct DecisionTreePlot {
    snarl: Snarl<Node>, // Define fields here
    viewer: DecisionTreeViewer,
}

impl DecisionTreePlot {
    fn new(cc: &eframe::CreationContext) -> Self {
        let snarl = Snarl::new();
        let viewer = DecisionTreeViewer;
        // Initialize fields here
        DecisionTreePlot { snarl, viewer }
    }
}

impl eframe::App for DecisionTreePlot {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            self.snarl
                .show(&mut self.viewer, &default_style(), "decision_tree", ui);
        });
    }
}

const fn default_style() -> SnarlStyle {
    SnarlStyle {
        node_layout: Some(NodeLayout::FlippedSandwich),
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
