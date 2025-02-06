use std::path::PathBuf;

pub fn full_file_path(relative_path: &str) -> PathBuf {
    let dir = format!(
        "{}/Work/ml-learn-rs/MachineLearningInActionSourceCode",
        dirs::home_dir().unwrap().to_str().unwrap()
    );

    PathBuf::from(dir).join(relative_path)
}

pub fn init_logs() {
    tracing_subscriber::fmt().init();
}
