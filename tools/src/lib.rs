use std::path::PathBuf;

pub fn full_file_path(relative_path: &str) -> PathBuf {
    let dir = format!(
        "{}/Work/ml-learn-rs/MachineLearningInActionSourceCode",
        dirs::home_dir().unwrap().to_str().unwrap()
    );

    PathBuf::from(dir).join(relative_path)
}

pub fn add_font(ctx: &egui::Context) {
    let mut font_definitions = egui::FontDefinitions::default();

    let font_data = egui::FontData::from_static(include_bytes!("../../assets/fonts/PingFang.ttc"));

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
