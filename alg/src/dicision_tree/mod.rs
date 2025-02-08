use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    hash::Hash,
};

use ndarray::{Array2, ArrayView2, Axis, s};

pub fn calculate_shannon_entropy<T: ToString>(data_set: ArrayView2<T>) -> f64 {
    let num_entries = data_set.shape()[0] as f64;

    let mut label_counts = std::collections::HashMap::new();

    for row in data_set.rows() {
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

pub fn split_data_set<T: Clone + Default + PartialEq>(
    data_set: ArrayView2<T>,
    index: usize,
    value: &T,
) -> Array2<T> {
    let mut ret_data_set = Array2::default([0, data_set.shape()[1] - 1]);

    for row in data_set.rows() {
        if &row[index] == value {
            let mut new_vec = row.to_owned();
            new_vec.remove_index(Axis(0), index);

            ret_data_set.push_row(new_vec.view()).unwrap();
        }
    }

    ret_data_set
}

pub fn choose_best_feature_to_split<T: Clone + Eq + Default + Display + Hash + ToString>(
    data_set: ArrayView2<T>,
) -> Option<usize> {
    // 最后一列当做label
    let num_features = data_set.shape()[1] - 1;

    let mut data = HashMap::new();

    let base_ent = calculate_shannon_entropy(data_set);
    tracing::info!("base entropy: {base_ent}");

    for i in 0..num_features {
        let column = data_set.column(i);
        let values: HashSet<_> = column.into_iter().collect();

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

#[cfg(test)]
mod tests {
    use ndarray::array;

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
        ];

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
        ];

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
        ];

        let idx = choose_best_feature_to_split(data_set.view());

        tracing::info!("best index: {idx:?}");
    }
}
