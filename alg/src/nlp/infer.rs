use ndarray::{Array1, Array2, Array3, Axis, stack};

pub fn word_id_to_one_hot(id: usize, vocab_size: usize) -> Array1<usize> {
    let mut one_hot = Array1::zeros(vocab_size);
    one_hot[id] = 1;
    one_hot
}

pub fn create_contexts_target(
    corpus: &[usize],
    window_size: usize,
    vocab_size: usize,
) -> (Array3<usize>, Array2<usize>) {
    let mut contexts = Array3::zeros((0, 2, vocab_size));
    let mut targets = Array2::zeros((0, vocab_size));

    for i in window_size..(corpus.len() - window_size - 1) {
        let c = stack![
            Axis(0),
            word_id_to_one_hot(corpus[i - 1], vocab_size),
            word_id_to_one_hot(corpus[i + 1], vocab_size),
        ];

        let t = word_id_to_one_hot(corpus[i], vocab_size);

        contexts.push(Axis(0), c.view()).unwrap();
        targets.push(Axis(0), t.view()).unwrap();
    }

    (contexts, targets)
}
