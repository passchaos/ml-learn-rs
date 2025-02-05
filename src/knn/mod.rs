use std::{
    collections::HashMap,
    io::{BufRead, BufReader},
};

use ndarray::{Array, Array1, Array2, ArrayView1, Axis};

pub fn classify(
    input: Array1<f64>,
    data_set: Array2<f64>,
    labels: Array1<String>,
    // sample_datas: &[SampleData],
    k: usize,
) -> String {
    let diff_mat = input - data_set;
    let sq_diff_mat = diff_mat.pow2();
    println!("diff mat: {diff_mat:?} {sq_diff_mat:?}");

    let sq_distances = sq_diff_mat.sum_axis(Axis(1));
    let distances = sq_distances.sqrt();
    println!("sq distances: {sq_distances:?} {distances:?}");

    let a = sq_distances[0];
    println!("a: {a}");

    let a = distances[0];
    println!("aa: {a}");

    let mut map: HashMap<_, usize> = HashMap::new();

    // argsort for rust
    let mut indices: Vec<_> = (0..distances.shape()[0]).collect();
    indices.sort_by(|a, b| distances[*a].total_cmp(&distances[*b]));

    for i in 0..k {
        let vote_label = &labels[indices[i]];
        *map.entry(vote_label).or_default() += 1;
    }

    println!("indices: {indices:?} map= {map:?}");

    map.into_iter().max_by_key(|ds| ds.1).unwrap().0.to_string()
}

pub fn file2matrix(file_path: &str) -> (Array2<f64>, Array1<String>) {
    let file = std::fs::File::open(file_path).unwrap();
    let file = BufReader::new(file);

    let mut arr = Array::zeros((0, 3));
    let mut labels = Array::default(0);

    for line in file.lines() {
        let line = line.unwrap();
        let mut data: Vec<&str> = line.split('\t').take(4).collect();

        let label = data.pop().unwrap().to_string();

        let data = Array1::from_vec(data).mapv(|a| a.trim().parse::<f64>().unwrap());
        let data = data.into_shape_with_order((1, 3)).unwrap();

        arr.append(Axis(0), data.view()).unwrap();
        labels
            .append(Axis(0), Array1::from(vec![label]).view())
            .unwrap();
    }

    (arr, labels)
}

pub fn auto_norm<'a>(data_set: ArrayView1<'a, f64>) -> (Array1<f64>, f64, f64) {
    let min_value = data_set.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let max_value = data_set.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

    let range = max_value - min_value;

    let res_set = data_set.iter().map(|v| (v - min_value) / range).collect();

    (res_set, range, *min_value)
}
