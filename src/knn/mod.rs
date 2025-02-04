use std::{
    collections::HashMap,
    io::{BufRead, BufReader},
};

use anyhow::Result;
use ndarray::{Array1, Array2, Axis};

use crate::{Data, SampleData};

pub fn classify(
    input: Array1<f64>,
    data_set: Array2<f64>,
    labels: Array1<String>,
    // sample_datas: &[SampleData],
    k: usize,
) -> String {
    let data_set_size = input.len_of(Axis(0));

    let diff_mat = input - data_set;
    let sq_diff_mat = diff_mat.pow2();
    println!("diff mat: {diff_mat:?} {sq_diff_mat:?}");

    let sq_distances = sq_diff_mat.sum_axis(Axis(1));
    let distances = sq_distances.sqrt();
    println!("sq distances: {sq_distances:?} {distances:?}");

    // let mut data_info: Vec<_> = sample_datas
    //     .iter()
    //     .map(|s_d| (s_d.data.distance(&input), s_d.label.to_string()))
    //     .collect();

    // println!("data info: {data_info:?}");
    // data_info.sort_by(|a, b| a.0.total_cmp(&b.0));
    // println!("sorted data info: {data_info:?}");

    // let mut results: HashMap<String, u32> = HashMap::new();

    // data_info.into_iter().take(k).for_each(|(_value, label)| {
    //     *results.entry(label).or_default() += 1;
    // });

    // let result = results
    //     .into_iter()
    //     .max_by_key(|ds| ds.1)
    //     .expect("can't get max result");

    "dd".to_string()
    // result.0
}

pub fn file2matrix(file_path: &str) -> Result<Vec<SampleData>> {
    let file = std::fs::File::open(file_path).unwrap();
    let file = BufReader::new(file);

    let mut result = vec![];

    for line in file.lines() {
        let line = line?;
        let mut data: Vec<_> = line.split('\t').take(4).collect();

        let label = data
            .pop()
            .ok_or(anyhow::anyhow!("line has no 4th content: {data:?}"))?
            .to_string();

        let mut inner = vec![];
        for d in data {
            let v = d.trim().parse()?;
            inner.push(v);
        }

        result.push(SampleData {
            data: Data { inner },
            label,
        });
    }

    Ok(result)
}

pub fn auto_norm(data_set: &[f64]) -> (Vec<f64>, f64, f64) {
    let min_value = *data_set.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let max_value = *data_set.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

    let range = max_value - min_value;

    let res_set = data_set.iter().map(|v| (v - min_value) / range).collect();

    (res_set, range, min_value)
}
