use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

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
    tracing::debug!("base entropy: {base_ent}");

    for i in 0..num_features {
        let axis_data = data_set.index_axis(Axis(1), i);
        let values: HashSet<_> = axis_data.into_iter().collect();

        let mut new_ent = 0.0;

        for value in values {
            let ret_data_set = split_data_set(data_set, i, value);

            let ent = calculate_shannon_entropy(ret_data_set.view());

            // ret_data_set.len()
            let value_probe = ret_data_set.len_of(Axis(0)) as f64 / data_set.len_of(Axis(0)) as f64;

            tracing::debug!(
                "index= {} value= {} value_probe= {value_probe} ent= {ent} ret= {}",
                i,
                value,
                ret_data_set
            );

            new_ent += value_probe * ent;
        }

        data.insert(i, new_ent);
    }

    tracing::debug!("all entropy: base= {base_ent} indexes= {data:?}");

    let (index, ent) = data.into_iter().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();

    if base_ent > ent {
        tracing::debug!("best feature index: {index} ent: {ent}");
        Some(index)
    } else {
        tracing::debug!("best feature index: -1 ent: {ent}");
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

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct MapValue {
    pub map: HashMap<String, MapValue>,
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
    features: &mut Array1<String>,
    map: &mut HashMap<String, MapValue>,
) {
    tracing::info!("data set: {data_set}");

    let labels = data_set.index_axis(Axis(1), data_set.len_of(Axis(1)) - 1);

    tracing::info!("labels: {labels}");

    // 类别都一样
    if labels.iter().filter(|a| **a == labels[0]).count() == labels.len() {
        let item = labels.into_iter().next().unwrap();
        map.insert(item.to_string(), MapValue::default());

        tracing::info!("final same");

        return;
    }

    if data_set.index_axis(Axis(0), 0).len() == 1 {
        let item = majority_cnt(labels);

        map.insert(item.to_string(), MapValue::default());

        tracing::info!("final majo");
        return;
    }

    // 排除没有熵减的情况
    let idx = choose_best_feature_to_split(data_set.clone()).unwrap();

    let values: HashSet<_> = data_set.index_axis(Axis(1), idx).into_iter().collect();

    let feature = features.get(idx).unwrap().clone();

    features.remove_index(Axis(0), idx);

    let mut outer_map = MapValue::default();

    for value in values {
        let data = split_data_set(data_set, idx, value);
        let view = data.view();

        tracing::info!(
            "data: index= {idx} value= {value} splited_data= {data} new_ref_data= {view}"
        );

        let mut inner_map = MapValue::default();

        create_tree(view, features, &mut inner_map.map);

        outer_map.map.insert(value.to_string(), inner_map);
    }

    map.insert(feature, outer_map);
}

fn classify(
    tree: &HashMap<String, MapValue>,
    features: ArrayView1<String>,
    data: ArrayView1<String>,
) -> String {
    assert_eq!(features.len(), data.len());

    let first_feature = tree.keys().next().unwrap();

    let f_idx = features.iter().position(|f| f == first_feature).unwrap();

    let value = &data[f_idx];

    let v = &tree[first_feature].map[value];

    let v_key = v.map.keys().next().unwrap();
    if v.map[v_key].map.is_empty() {
        // 已经是叶子节点了
        return v_key.to_string();
    } else {
        return classify(&v.map, features.clone(), data);
    }
}

fn tree_to_dot_content_impl(tree: &HashMap<String, MapValue>, c: &mut String) {
    for (feature, value) in tree {
        for (inner_feature, inner_value) in &value.map {
            for (inner_inner_feature, inner_inner_value) in &inner_value.map {
                let len = c.len();

                let has_next_level = !inner_inner_value.map.is_empty();

                if has_next_level {
                    c.push_str(&format!(
                        "   \"{feature}\" -> \"{inner_inner_feature}\" [label = \"{inner_feature}\"];\n"
                    ));

                    tree_to_dot_content_impl(&inner_value.map, c);
                } else {
                    let target = format!("target_{len}");

                    c.push_str(&format!(
                        "   {target} [label = \"{inner_inner_feature}\"];\n"
                    ));
                    c.push_str(&format!(
                        "   \"{feature}\" -> {target} [label = \"{inner_feature}\"];\n"
                    ));
                }
            }
        }
    }
}

pub fn tree_to_dot_content(tree: &HashMap<String, MapValue>) -> String {
    let mut c = String::from("digraph Tree {\n");

    tree_to_dot_content_impl(tree, &mut c);

    c.push_str("\n}");

    c
}

#[cfg(test)]
mod tests {
    use ndarray::{array, s};

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
    fn test_create_dicision_tree_1() {
        tracing_subscriber::fmt().init();

        let data_set = array![
            ["1", "1", "yes"],
            ["1", "1", "yes"],
            ["1", "0", "no"],
            ["0", "1", "no"],
            ["0", "1", "no"],
        ]
        .map(|a| a.to_string());

        let mut map = HashMap::new();

        let features = array!["no surfacing", "flippers"].mapv(|a| a.to_string());

        create_tree(data_set.view(), &mut features.clone(), &mut map);
        tracing::info!("map: {map:#?}");

        let dot_c = tree_to_dot_content(&map);
        tracing::info!("dot: {dot_c}");

        let data = array!["1", "1"].map(|a| a.to_string());

        let feature = classify(&map, features.view(), data.view());
        tracing::info!("classified feature: {feature}");
    }

    #[test]
    fn test_create_dicision_tree_2() {
        tracing_subscriber::fmt().init();

        let data_set = array![
            ["1", "1", "yes"],
            ["1", "1", "yes"],
            ["1", "0", "no"],
            ["0", "1", "no"],
            ["0", "1", "yes"],
        ]
        .map(|a| a.to_string());

        let mut map = HashMap::new();

        let mut features = array!["no surfacing", "flippers"].mapv(|a| a.to_string());

        create_tree(data_set.view(), &mut features, &mut map);
        tracing::info!("map: {map:#?}");
    }
}
