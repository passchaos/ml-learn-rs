use std::io::Cursor;

use egui::{ColorImage, Image, ImageSource};
use ndarray::{Array2, Axis};

fn main() {
    let mnist_dir = std::env::home_dir().unwrap().join("Work/mnist");

    let train_data_path = mnist_dir.join("train-images-idx3-ubyte");
    let data = alg::dataset::mnist::load_images(train_data_path);

    eframe::run_native(
        "mnist",
        eframe::NativeOptions::default(),
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);

            Ok(Box::new(App { data, idx: 0 }))
        }),
    )
    .unwrap();
}

struct App {
    data: Array2<u8>,
    idx: usize,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let data_0_len = self.data.len_of(Axis(0));

            if ui.button("--->").clicked() {
                self.idx = (self.idx + data_0_len - 1) % data_0_len;
            }

            let img_data = self.data.index_axis(Axis(0), self.idx).to_vec();

            let image = ColorImage::from_gray([28, 28], &img_data);
            let texture = ctx.load_texture("abc", image, egui::TextureOptions::default());

            let img = Image::new(&texture).shrink_to_fit();
            ui.add_sized([300.0, 300.0], img);

            if ui.button("<---").clicked() {
                self.idx = (self.idx + 1) % data_0_len;
            }

            // let image = ImageSource::Bytes {
            //     uri: "test.png".into(),
            //     bytes: img_data.into(),
            // };
            // ui.image(image);
        });
    }
}
