use std::collections::HashMap;

use crate::{Data, Label, SampleData};

pub fn classify(input: Data, sample_datas: &[SampleData], k: usize) -> Label {
    let mut data_info: Vec<_> = sample_datas
        .iter()
        .map(|s_d| (s_d.data.distance(&input), s_d.label))
        .collect();

    println!("data info: {data_info:?}");
    data_info.sort_by(|a, b| a.0.total_cmp(&b.0));
    println!("sorted data info: {data_info:?}");

    let mut results: HashMap<Label, u32> = HashMap::new();

    data_info.into_iter().take(k).for_each(|(_value, label)| {
        *results.entry(label).or_default() += 1;
    });

    let result = results
        .into_iter()
        .max_by_key(|ds| ds.1)
        .expect("can't get max result");

    result.0
}
