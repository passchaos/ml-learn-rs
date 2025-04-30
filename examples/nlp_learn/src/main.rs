extern crate openblas_src;

use std::{collections::HashMap, f32};

use ndarray::{Array, Array1, Array2, ArrayView2, Axis};
use ndarray_linalg::SVD;

fn preprocess(text: &str) -> (Vec<usize>, HashMap<String, usize>, HashMap<usize, String>) {
    let words = text.to_lowercase();
    let words = words.replace('.', " .");
    let words = words.split_whitespace();

    let mut corpus = Vec::new();
    let mut word_to_id = HashMap::new();
    let mut id_to_word = HashMap::new();

    for word in words.clone() {
        if !word_to_id.contains_key(word) {
            let new_id = word_to_id.len();

            word_to_id.insert(word.to_string(), new_id);
            id_to_word.insert(new_id, word.to_string());
        }
    }

    for word in words {
        let idx = word_to_id.get(word).unwrap();
        corpus.push(*idx);
    }

    // let corpus = words.map(|a| a.to_string()).collect();

    (corpus, word_to_id, id_to_word)
}

fn create_co_matrix(corpus: &[usize], vocab_size: usize, window_size: usize) -> Array2<usize> {
    let corpus_size = corpus.len();

    let mut co_matrix = Array2::zeros((vocab_size, vocab_size));

    for (idx, word_id) in corpus.iter().enumerate() {
        for i in 1..(window_size + 1) {
            let idx = idx as isize;
            let i = i as isize;

            let left_idx = idx - i;
            let right_idx = idx + i;

            if left_idx >= 0 {
                let left_word_id = corpus[left_idx as usize];

                co_matrix[[*word_id, left_word_id]] += 1;
            }

            if right_idx < corpus_size as isize {
                let right_word_id = corpus[right_idx as usize];

                co_matrix[[*word_id, right_word_id]] += 1;
            }
        }
    }

    co_matrix
}

fn most_similar(
    query: &str,
    word_to_id: &HashMap<String, usize>,
    id_to_word: &HashMap<usize, String>,
    word_matrix: &Array2<usize>,
) -> Array1<(String, f32)> {
    let query_id = word_to_id[query];
    let query_vec = word_matrix.index_axis(Axis(0), query_id);

    let vocab_size = word_to_id.len();

    let mut arr: Vec<_> = (0..vocab_size)
        .into_iter()
        .filter_map(|idx| {
            if idx == query_id {
                None
            } else {
                let word_id = idx;
                let word_vec = word_matrix.index_axis(Axis(0), word_id);

                let word = id_to_word[&word_id].clone();
                let similarity = alg::math::cos_similarity(
                    &query_vec.mapv(|a| a as f32).view(),
                    &word_vec.mapv(|a| a as f32).view(),
                );
                Some((word, similarity))
            }
        })
        .collect();
    arr.sort_by(|a, b| b.1.total_cmp(&a.1));

    Array1::from_vec(arr)
}

fn ppmi(c: &ArrayView2<usize>) -> Array2<f32> {
    let mut m = Array::<f32, _>::zeros(c.raw_dim());

    let n = c.sum();
    let s = c.sum_axis(Axis(0));

    let (h, w) = (c.shape()[0], c.shape()[1]);

    for i in 0..h {
        for j in 0..w {
            let pmi = ((c[(i, j)] * n) as f32 / ((s[i] * s[j]) as f32 + f32::EPSILON)).log2();

            m[[i, j]] = pmi.max(0.0);
        }
    }

    m
}

fn main() {
    let text = "You say goodbye and I say hello.";
    let (corpus, word_to_id, id_to_word) = preprocess(text);

    let co_matrix = create_co_matrix(&corpus, word_to_id.len(), 1);

    let c0 = co_matrix.index_axis(Axis(0), word_to_id["you"]);
    let c1 = co_matrix.index_axis(Axis(0), word_to_id["i"]);

    let simil =
        alg::math::cos_similarity(&c0.mapv(|a| a as f32).view(), &c1.mapv(|a| a as f32).view());
    let ppmi = ppmi(&co_matrix.view());
    println!("co matrix: {co_matrix} ppmi= {ppmi} simil= {simil}");

    let res = most_similar("you", &word_to_id, &id_to_word, &co_matrix);
    println!("most similar: res= {res:?}");

    let (u, s, vt) = ppmi.svd(true, true).unwrap();

    let u = u.unwrap();
    let vt = vt.unwrap();
    println!("u: {u} s: {s} vt: {vt}");
}
