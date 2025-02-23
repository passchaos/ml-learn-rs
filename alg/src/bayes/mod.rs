use std::collections::HashSet;

use ndarray::{Array1, ArrayView1, ArrayView2, Axis, s};

#[allow(dead_code)]
fn load_data_set() -> (Vec<u32>, Vec<Vec<String>>) {
    let data = vec![
        vec![
            "my", "dog", "has", "has", "flea", "problems", "help", "please",
        ],
        vec!["maybe", "not", "take", "him", "to", "dog", "park", "stupid"],
        vec!["my", "dalmation", "is", "so", "cute", "I", "love", "him"],
        vec!["stop", "posting", "stupid", "worthless", "garbage"],
        vec![
            "mr", "licks", "ate", "my", "steak", "how", "to", "stop", "him",
        ],
        vec!["quit", "buying", "worthless", "dog", "food", "stupid"],
    ]
    .into_iter()
    .map(|a| a.into_iter().map(|a| a.to_string()).collect())
    .collect();

    (vec![0, 1, 0, 1, 0, 1], data)
}

pub fn create_vocab_list(data_set: Vec<Vec<String>>) -> Vec<String> {
    let data: HashSet<_> = data_set.into_iter().flatten().collect();

    data.into_iter().collect()
}

pub fn words_set_to_vec(vocab_list: &[String], input_set: &[String]) -> Array1<u32> {
    let mut vec = ndarray::Array1::zeros(vocab_list.len());

    for word in input_set {
        if let Some(idx) = vocab_list.iter().position(|a| a == word) {
            vec[idx] += 1;
        }
    }

    vec
}

pub fn train_naive_bayes_0(
    train_matrix: ArrayView2<u32>,
    train_category: ArrayView1<u32>,
) -> (Array1<f32>, Array1<f32>, f32) {
    let num_train_docs = train_matrix.len_of(Axis(0));
    let num_words = train_matrix.len_of(Axis(1));

    let p_abusive = train_category.sum() as f32 / num_train_docs as f32;
    let mut p0_num: Array1<f32> = Array1::ones(num_words);
    let mut p1_num: Array1<f32> = Array1::ones(num_words);

    let mut p0_denom = 2.0;
    let mut p1_denom = 2.0;

    for i in 0..num_train_docs {
        let data = train_matrix.slice(s![i, ..]).map(|a| *a as f32);

        if train_category[i] == 1 {
            p1_num = p1_num + &data;
            p1_denom += data.sum();
        } else {
            p0_num = p0_num + &data;
            p0_denom += data.sum();
        }
    }

    let p1_vect = (p1_num / p1_denom).ln();
    let p0_vect = (p0_num / p0_denom).ln();

    (p0_vect, p1_vect, p_abusive)
}

pub fn classify_naive_bayes(
    data_vec: ArrayView1<u32>,
    p0_vec: ArrayView1<f32>,
    p1_vec: ArrayView1<f32>,
    p_abusive: f32,
) -> u32 {
    let data = data_vec.map(|a| *a as f32);
    let p1 = (&data * &p1_vec).sum() + p_abusive.ln();
    let p0 = (&data * &p0_vec).sum() + (1.0 - p_abusive).ln();

    if p1 > p0 { 1 } else { 0 }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;

    #[test]
    fn test_bayes_data_set() {
        let (labels, data_set) = load_data_set();
        let vocab_list = create_vocab_list(data_set.clone());

        println!("vocab list: {vocab_list:?}");

        // let input1 = data_set.iter().map(|a| a.1.clone()).nth(0).unwrap();
        // let vec1 = words_set_to_vec(&vocab_list, &input1);
        let labels = Array1::from_vec(labels);

        let mut train_mat = Array2::zeros([0, vocab_list.len()]);

        for vec in data_set.into_iter() {
            let vec = words_set_to_vec(&vocab_list, &vec);

            train_mat.push(Axis(0), vec.view()).unwrap();
        }
        // println!("vec1: {vec1:?}");
        println!("labels: {labels:?} train_mat= {train_mat:?}");

        let (p0_vect, p1_vect, p_abusive) = train_naive_bayes_0(train_mat.view(), labels.view());
        println!("p0_vect: {p0_vect:?} p1_vect: {p1_vect:?} p_abusive: {p_abusive}");

        let test_input = vec![vec!["love", "my", "dalmation"], vec!["stupid", "garbage"]]
            .into_iter()
            .map(|a| a.into_iter().map(|a| a.to_string()).collect::<Vec<_>>());

        for input in test_input {
            let test_doc = words_set_to_vec(&vocab_list, &input);
            let result =
                classify_naive_bayes(test_doc.view(), p0_vect.view(), p1_vect.view(), p_abusive);
            println!("classify: input= {input:?} result= {result}");
        }
    }
}
