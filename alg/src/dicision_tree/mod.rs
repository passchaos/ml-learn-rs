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
    axis: Axis,
    value: T,
) -> Array2<T> {
    let mut ret_data_set = Array2::default([0, data_set.shape()[1] - 1]);

    for row in data_set.rows() {
        if row[axis.index()] == value {
            let mut new_vec = row.to_owned();
            new_vec.remove_index(Axis(0), axis.index());

            ret_data_set.push_row(new_vec.view()).unwrap();
        }
    }

    ret_data_set
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

        let ret_data_set = split_data_set(data_set.view(), Axis(0), "1");

        tracing::info!("ret_data_set: {ret_data_set}");
    }
}
