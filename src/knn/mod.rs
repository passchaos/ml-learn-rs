use std::{
    collections::HashMap,
    io::{BufRead, BufReader},
    path::Path,
};

use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Axis};

pub fn classify(
    input: ArrayView1<f64>,
    data_set: ArrayView2<f64>,
    labels: ArrayView1<String>,
    // sample_datas: &[SampleData],
    k: usize,
) -> String {
    let diff_mat = &input - &data_set;
    let sq_diff_mat = diff_mat.pow2();
    tracing::debug!("diff mat: {diff_mat:?} {sq_diff_mat:?}");

    let sq_distances = sq_diff_mat.sum_axis(Axis(1));
    let distances = sq_distances.sqrt();
    tracing::debug!("sq distances: {sq_distances:?} {distances:?}");

    let a = sq_distances[0];
    tracing::debug!("a: {a}");

    let a = distances[0];
    tracing::debug!("aa: {a}");

    let mut map: HashMap<_, usize> = HashMap::new();

    // argsort for rust
    let mut indices: Vec<_> = (0..distances.shape()[0]).collect();
    indices.sort_by(|a, b| distances[*a].total_cmp(&distances[*b]));

    for i in 0..k {
        let vote_label = &labels[indices[i]];
        *map.entry(vote_label).or_default() += 1;
    }

    tracing::debug!("indices: {indices:?} map= {map:?}");

    map.into_iter().max_by_key(|ds| ds.1).unwrap().0.to_string()
}

pub fn file2matrix<P: AsRef<Path>>(file_path: P) -> (Array2<f64>, Array1<String>) {
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

pub fn auto_norm(data_set: ArrayView2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let min_vals = data_set.fold_axis(Axis(0), f64::INFINITY, |a, b| a.min(*b));
    let max_vals = data_set.fold_axis(Axis(0), f64::NEG_INFINITY, |a, b| a.max(*b));

    let range = max_vals - &min_vals;

    let res_set = (&data_set - &min_vals.view()) / &range;

    (res_set, range, min_vals)
}

#[cfg(test)]
mod tests {
    use ndarray::{array, s};

    use super::*;

    #[test]
    fn test_dating_classify() {
        crate::tools::init_logs();

        let ho_ratio = 0.1;

        let (data_set, labels) =
            file2matrix(crate::tools::full_file_path("Ch02/datingTestSet.txt"));
        let (norm_data_set, _, _) = auto_norm(data_set.view());

        let m = norm_data_set.shape()[0];

        let mut err_count = 0;

        let num_test_vecs = (m as f64 * ho_ratio) as usize;

        for i in 0..num_test_vecs {
            let classifier_result = classify(
                norm_data_set.slice(s![i, ..]),
                norm_data_set.slice(s![num_test_vecs..m, ..]),
                labels.slice(s![num_test_vecs..m]),
                3,
            );
            tracing::info!(
                "the classifier came back with: {} the real answer is: {}",
                classifier_result,
                labels[i]
            );

            if classifier_result != labels[i] {
                tracing::warn!("meet error: err_count= {err_count}");
                err_count += 1;
            }
        }

        tracing::info!(
            "the total error rate is {}, err_count= {err_count} num_test= {num_test_vecs}",
            err_count as f32 / num_test_vecs as f32
        );

        // let test_data = Array1::from_vec(vec![40920.0, 8.326976, 0.953952]);
        // let res = classify(test_data, norm_data_set, labels, 3);
        // println!("res: {res}");
    }

    #[test]
    fn test_classify_person() {
        crate::tools::init_logs();

        let result_list = ["not at all", "in small doses", "in large doses"];

        let mut input = String::new();

        print!("percentage of time spent playing video game?");
        std::io::stdin().read_line(&mut input).unwrap();
        let percent_tats: f64 = input.parse().unwrap();

        input = String::new();
        print!("frequent flier miles earned per year?");
        std::io::stdin().read_line(&mut input).unwrap();
        let ff_miles: f64 = input.parse().unwrap();

        input = String::new();
        print!("liters of ice cream consumed per year?");
        std::io::stdin().read_line(&mut input).unwrap();
        let ice_cream: f64 = input.parse().unwrap();

        let (dating_data_mat, dating_labels) =
            file2matrix(crate::tools::full_file_path("Ch02/datingTestSet2.txt"));

        let (norm_mat, ranges, min_vals) = auto_norm(dating_data_mat.view());

        let in_arr = (array![ff_miles, percent_tats, ice_cream] - min_vals) / ranges;

        let classifier_result = classify(in_arr.view(), norm_mat.view(), dating_labels.view(), 3);

        let v: usize = classifier_result.parse().unwrap();
        println!("You will probably like this person: {}", result_list[v - 1]);
    }
}
