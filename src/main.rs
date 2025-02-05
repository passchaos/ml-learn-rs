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
    let b = data.slice(s![.., 1]);
    let c = data.slice(s![.., 2]);

    let weight: Array1<u8> = labels.iter().map(|a| a.parse().unwrap()).collect();

    let data = PlotDemo {
        a,
        b,
        c,
        weight,
        relation: Relation::AB,
    };

    eframe::run_native(
        "Plot",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Ok(Box::new(data))),
    )
    .unwrap();

    // let licheng_values: Vec<_> = samples.iter().map(|sd| sd.data.inner[0]).collect();
    // let (licheng_normed_values, licheng_range, licheng_min) = knn::auto_norm(&licheng_values);
    // println!("normed values= {licheng_normed_values:?} range= {licheng_range} min= {licheng_min}");

    // let game_values: Vec<_> = samples.iter().map(|sd| sd.data.inner[1]).collect();
    // let (game_normed_values, game_range, game_min) = knn::auto_norm(&game_values);
    // let ice_cream_values: Vec<_> = samples.iter().map(|sd| sd.data.inner[2]).collect();
    // let (ice_normed_values, ice_range, ice_min) = knn::auto_norm(&ice_cream_values);

    // let labels: Vec<usize> = samples.iter().map(|sd| sd.label.parse().unwrap()).collect();

    // let size_arr: Vec<_> = labels.iter().map(|ll| ll * 5).collect();
    // let marker = Marker::new()
    //     .size_array(size_arr.clone())
    //     .color_array(size_arr);

    // let trace1 = Scatter::new(licheng_normed_values, game_normed_values)
    //     .mode(Mode::Markers)
    //     .marker(marker);

    // let layout = Layout::new()
    //     .x_axis(Axis::new().title("每年获取的飞行里程数"))
    //     .y_axis(Axis::new().title("玩视频游戏所耗时间百分比"))
    //     .legend(Legend::new());

    // let mut plt = Plot::new();
    // plt.add_trace(trace1);
    // plt.set_layout(layout);
    // plt.show();

    Ok(())
}

#[derive(PartialEq)]
enum Relation {
    AB,
    AC,
    BC,
}

struct PlotDemo<'a> {
    a: ArrayView1<'a, f64>,
    b: ArrayView1<'a, f64>,
    c: ArrayView1<'a, f64>,
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

        egui::CentralPanel::default().show(ctx, |ui| {
            egui_plot::Plot::new("plot").show(ui, |plot_ui| {
                let (x, y) = match self.relation {
                    Relation::AB => (self.a, self.b),
                    Relation::AC => (self.a, self.c),
                    Relation::BC => (self.b, self.c),
                };

                let data = x
                    .iter()
                    .zip(y.iter())
                    .map(|(a, b)| [*a, *b])
                    .zip(self.weight.iter());

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
