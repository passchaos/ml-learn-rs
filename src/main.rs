use anyhow::Result;
use plotly::{
    Plot, Scatter,
    common::{Marker, Mode},
};

mod knn;

#[derive(Debug)]
struct Data {
    inner: Vec<f64>,
}

impl Data {
    fn distance(&self, other: &Data) -> f64 {
        let value: f64 = self
            .inner
            .iter()
            .zip(other.inner.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        value.sqrt()
    }
}

#[derive(Debug)]
struct SampleData {
    data: Data,
    label: String,
}

fn main() -> Result<()> {
    let sample_datas: Vec<_> = vec![
        (vec![1.0, 1.1], "A"),
        (vec![1.0, 1.0], "A"),
        (vec![0.0, 0.0], "B"),
        (vec![0.0, 0.1], "B"),
    ]
    .into_iter()
    .map(|(inner, label)| SampleData {
        data: Data { inner },
        label: label.to_string(),
    })
    .collect();

    let input = Data {
        inner: vec![0.0, 0.0],
    };
    let res = knn::classify(input, &sample_datas, 3);
    println!("result: {res:?}");

    let file_path = format!(
        "{}/Work/MachineLearningInActionSourceCode/Ch02/datingTestSet2.txt",
        dirs::home_dir()
            .and_then(|hd| hd.to_str().map(|s| s.to_string()))
            .ok_or(anyhow::anyhow!("can't get home dir"))?
    );

    let samples = knn::file2matrix(&file_path)?;
    println!("samples: {samples:?}");

    let licheng_values: Vec<_> = samples.iter().map(|sd| sd.data.inner[0]).collect();
    let game_values: Vec<_> = samples.iter().map(|sd| sd.data.inner[1]).collect();
    let ice_cream_values: Vec<_> = samples.iter().map(|sd| sd.data.inner[2]).collect();
    let labels: Vec<usize> = samples.iter().map(|sd| sd.label.parse().unwrap()).collect();

    let size_arr: Vec<_> = labels.iter().map(|ll| ll * 5).collect();
    let marker = Marker::new()
        .size_array(size_arr.clone())
        .color_array(size_arr);

    let trace1 = Scatter::new(licheng_values, game_values)
        .mode(Mode::Markers)
        .marker(marker);

    let mut plt = Plot::new();
    plt.add_trace(trace1);
    plt.show();

    Ok(())
}
