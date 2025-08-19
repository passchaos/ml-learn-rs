use std::path::PathBuf;

use anyhow::Result;

pub fn init_log() {
    let format = tracing_subscriber::fmt::format()
        .with_level(true)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_file(true)
        .with_source_location(true);

    tracing_subscriber::fmt().event_format(format).init();
}

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
