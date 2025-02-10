use std::{
    collections::{HashMap, HashSet},
    fmt::{Debug, Display},
    hash::Hash,
};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, IndexLonger, s};

pub fn calculate_shannon_entropy(data_set: ArrayView2<String>) -> f64 {
    let num_entries = data_set.len_of(Axis(0)) as f64;

    let mut label_counts = std::collections::HashMap::new();

    for row in data_set.axis_iter(Axis(0)) {
        let label = row[row.len() - 1].to_string();
        *label_counts.entry(label).or_insert(0) += 1;
    }

    let mut shannon_entropy = 0.0;

    for (_, count) in label_counts {
        let prob = count as f64 / num_entries;
        shannon_entropy -= prob * prob.log2();
    }

    shannon_entropy
}

pub fn split_data_set(data_set: ArrayView2<String>, index: usize, value: &str) -> Array2<String> {
    let mut ret_data_set = Array2::default([0, data_set.shape()[1] - 1]);

    for row in data_set.axis_iter(Axis(0)) {
        if &row[index] == value {
            let mut new_vec = row.to_owned();
            new_vec.remove_index(Axis(0), index);

            ret_data_set.push(Axis(0), new_vec.view()).unwrap();
        }
    }

    ret_data_set
}

pub fn choose_best_feature_to_split(data_set: ArrayView2<String>) -> Option<usize> {
    // 最后一列当做label
    let num_features = data_set.shape()[1] - 1;

    let mut data = HashMap::new();

    let base_ent = calculate_shannon_entropy(data_set);
    tracing::info!("base entropy: {base_ent}");

    for i in 0..num_features {
        let axis_data = data_set.index_axis(Axis(1), i);
        let values: HashSet<_> = axis_data.into_iter().collect();

        let mut new_ent = 0.0;

        for value in values {
            let ret_data_set = split_data_set(data_set, i, value);

            let ent = calculate_shannon_entropy(ret_data_set.view());

            // ret_data_set.len()
            let value_probe = ret_data_set.len_of(Axis(0)) as f64 / data_set.len_of(Axis(0)) as f64;

            tracing::info!(
                "index= {} value= {} value_probe= {value_probe} ent= {ent} ret= {}",
                i,
                value,
                ret_data_set
            );

            new_ent += value_probe * ent;
        }

        data.insert(i, new_ent);
    }

    tracing::info!("all entropy: base= {base_ent} indexes= {data:?}");

    let (index, ent) = data.into_iter().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();

    if base_ent > ent {
        tracing::info!("best feature index: {index} ent: {ent}");
        Some(index)
    } else {
        tracing::info!("best feature index: -1 ent: {ent}");
        None
    }
}

pub fn majority_cnt(class_list: ArrayView1<String>) -> String {
    let mut map = HashMap::new();

    for class in class_list.iter() {
        *map.entry(class).or_insert(0) += 1;
    }

    let value = map.into_iter().max_by_key(|(_a, b)| *b).unwrap();
    value.0.to_owned()
}

#[derive(Debug)]
struct MapValue {
    map: HashMap<String, MapValue>,
}

impl From<HashMap<String, MapValue>> for MapValue {
    fn from(map: HashMap<String, MapValue>) -> Self {
        Self { map }
    }
}

// data_set是2维数组，每一行是一个样本，最后一列是label
// property是1维数组，每一个元素是一个特征，其长度等于data_set的列数-1
pub fn create_tree(
    data_set: ArrayView2<String>,
    features: ArrayView1<String>,
) -> HashMap<String, MapValue> {
    // 排除没有熵减的情况
    let idx = choose_best_feature_to_split(data_set.clone()).unwrap();

    // data_set
    // data_set.slice(info)

    // split_data_set(data_set, idx, )
    let mut map = HashMap::new();

    let mut inner_map = HashMap::new();

    let value = MapValue { map: inner_map };

    map.insert("top".to_string(), value);

    map
}

#[cfg(test)]
mod tests {
    use ndarray::{ArrayBase, array};

    use super::*;

    #[test]
    fn test_shannno_entropy() {
        tracing_subscriber::fmt().init();

        let data_set = array![
            ["1", "1", "yes"],
            ["1", "1", "yes"],
            ["1", "0", "no"],
            ["0", "1", "no"],
            ["0", "1", "no"],
            ["2", "2", "es"]
        ]
        .mapv(|a| a.to_string());

        let shannon_entropy = calculate_shannon_entropy(data_set.view());
        tracing::info!("entropy: {shannon_entropy}");

        // assert_eq!(shannon_entropy, 0.9709505944546686);
    }

    #[test]
    fn test_split_data_set() {
        tracing_subscriber::fmt().init();

        let data_set = array![
            ["1", "1", "yes"],
            ["1", "1", "yes"],
            ["1", "0", "no"],
            ["0", "1", "no"],
            ["0", "1", "no"],
            ["2", "2", "es"]
        ]
        .mapv(|a| a.to_string());

        let ret_data_set = split_data_set(data_set.view(), 0, &"1");

        tracing::info!("ret_data_set: {ret_data_set}");

        tracing::info!("axis 0: {}", data_set.slice(s![0, ..]));
    }

    #[test]
    fn test_feature_split() {
        tracing_subscriber::fmt().init();

        let data_set = array![
            ["1", "1", "yes"],
            ["1", "1", "yes"],
            ["1", "0", "no"],
            ["0", "1", "no"],
            ["0", "1", "no"],
        ]
        .mapv(|a| a.to_string());

        let idx = choose_best_feature_to_split(data_set.view());

        tracing::info!("best index: {idx:?}");
    }

    #[test]
    fn test_create_dicision_tree() {
        tracing_subscriber::fmt().init();

        let data_set = array![
            ["1", "1", "yes"],
            ["1", "1", "yes"],
            ["1", "0", "no"],
            ["0", "1", "no"],
            ["0", "1", "no"],
        ]
        .map(|a| a.to_string());

        let map = create_tree(data_set.view(), array![].view());
        tracing::info!("map: {map:?}");

        let a = array![[1, 2, 3], [4, 5, 6]];

        for row in a.rows() {
            tracing::info!("row: {row}");
        }

        let idx = 0;
        for row in a.axis_iter(Axis(idx)) {
            tracing::info!("axis {idx}: {row}");
        }

        for lane in a.lanes(Axis(idx)) {
            tracing::info!("lane {idx}: {lane}");
        }

        for c in a.columns() {
            tracing::info!("column: {c}");
        }
    }
}
