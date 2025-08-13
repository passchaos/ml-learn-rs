use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayView2, Axis};

type Float = f64;

pub fn preprocess(text: &str) -> (Vec<usize>, HashMap<String, usize>, HashMap<usize, String>) {
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

pub fn create_co_matrix(corpus: &[usize], vocab_size: usize, window_size: usize) -> Array2<Float> {
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

                co_matrix[[*word_id, left_word_id]] += 1.0;
            }

            if right_idx < corpus_size as isize {
                let right_word_id = corpus[right_idx as usize];

                co_matrix[[*word_id, right_word_id]] += 1.0;
            }
        }
    }

    co_matrix
}

pub fn most_similar(
    query: &str,
    word_to_id: &HashMap<String, usize>,
    id_to_word: &HashMap<usize, String>,
    word_matrix: &ArrayView2<Float>,
    top: usize,
) -> Array1<(String, Float)> {
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
                let similarity = crate::math::cos_similarity(
                    &query_vec.mapv(|a| a).view(),
                    &word_vec.mapv(|a| a).view(),
                );
                Some((word, similarity))
            }
        })
        .collect();
    arr.sort_by(|a, b| b.1.total_cmp(&a.1));

    Array1::from_iter(arr.into_iter().take(top))
}

pub fn ppmi(c: &ArrayView2<Float>) -> Array2<Float> {
    let mut m = Array2::zeros(c.raw_dim());

    let n = c.sum();
    let s = c.sum_axis(Axis(0));

    let (h, w) = (c.shape()[0], c.shape()[1]);

    for i in 0..h {
        for j in 0..w {
            let pmi = ((c[(i, j)] * n) / (s[i] * s[j] + Float::EPSILON)).log2();

            m[[i, j]] = pmi.max(0.0);
        }
    }

    m
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use ndarray::s;
    use ndarray_linalg::{SVD, TruncatedOrder, TruncatedSvd, svd};

    use super::*;

    #[test]
    fn test_preprocess() {
        let text = "You say goodbye and I say hello.";
        let (corpus, word_to_id, id_to_word) = preprocess(text);
        println!("corpus: {corpus:?}");
        println!("word_to_id: {word_to_id:?}");
        println!("id_to_word: {id_to_word:?}");
        let co_matrix = create_co_matrix(&corpus, word_to_id.len(), 1);
        println!("co_matrix: {co_matrix}");

        let similar_rates = most_similar("you", &word_to_id, &id_to_word, &co_matrix.view(), 5);
        println!("similar rates: {similar_rates:?}");

        let ppmi = ppmi(&co_matrix.view());
        println!("ppmi: {ppmi:?}");

        let (u, s, vt) = ppmi.svd(true, true).unwrap();
        let s = Array2::from_diag(&s);
        println!("u: {u:?}");
        println!("s: {s:?}");
        println!("vt: {vt:?}");
    }

    #[test]
    fn test_ptb_dataset() {
        let path = std::env::home_dir()
            .unwrap()
            .join("Downloads/ptb/ptb.train.txt");
        let f = std::fs::read_to_string(path).unwrap();
        let (corpus, word_to_id, id_to_word) = preprocess(&f);

        let c = create_co_matrix(&corpus, word_to_id.len(), 2);
        let ppmi = ppmi(&c.view());

        // {
        //     let begin = Instant::now();
        //     let (u, s, vt) = ppmi.svd(true, true).unwrap();
        //     let elapsed = begin.elapsed();

        //     println!("svd s: {s} elapsed= {}", elapsed.as_millis());
        // }

        let begin = Instant::now();
        let res = TruncatedSvd::new(ppmi, TruncatedOrder::Largest)
            .decompose(100)
            .unwrap();

        let elapsed = begin.elapsed();
        let (u, s, vt) = res.values_vectors();
        println!("truncated svd s: {s:?} elapsed= {}", elapsed.as_millis());

        let word_vecs = u.slice(s![.., 0..100]);

        let querys = ["you", "year", "car", "toyota"];
        for query in querys {
            let similar_rates = most_similar(query, &word_to_id, &id_to_word, &word_vecs, 5);
            println!("similar rates for {query}: {similar_rates:?}");
        }
    }
}
