use std::path::PathBuf;

use anyhow::Result;

pub fn md5_hex(input: impl AsRef<[u8]>) -> String {
    use md5::Digest;

    let mut hasher = md5::Md5::new();
    hasher.update(input);
    format!("{:x}", hasher.finalize())
}

pub fn dot_to_svg(dot_s: &str) -> Result<String> {
    let mut parser = layout::gv::DotParser::new(&dot_s);

    let g = parser.process().map_err(|e| anyhow::anyhow!(e))?;

    let mut g_b = layout::gv::GraphBuilder::new();
    g_b.visit_graph(&g);
    let mut vg = g_b.get();

    let mut sw = layout::backends::svg::SVGWriter::new();
    vg.do_it(false, false, false, &mut sw);

    Ok(sw.finalize())
}

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

    font_definitions
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .insert(0, "pingfang".to_string());

    ctx.set_fonts(font_definitions);
}
