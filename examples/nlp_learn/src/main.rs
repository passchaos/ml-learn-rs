extern crate openblas_src;

use std::collections::HashMap;

use ndarray::{Array2, Axis};

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

fn main() {
    let text = "You say goodbye and I say hello.";
    let (corpus, word_to_id, id_to_word) = preprocess(text);

    let co_matrix = create_co_matrix(&corpus, word_to_id.len(), 1);

    let c0 = co_matrix.index_axis(Axis(0), word_to_id["you"]);
    let c1 = co_matrix.index_axis(Axis(0), word_to_id["i"]);

    let simil =
        alg::math::cos_similarity(&c0.mapv(|a| a as f32).view(), &c1.mapv(|a| a as f32).view());
    println!("co matrix: {co_matrix} simil= {simil}");
}
