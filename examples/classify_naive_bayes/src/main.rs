use ndarray::{Array1, Array2, Axis};
use rand::Rng;

fn text_parse(input: &str) -> Vec<String> {
    let re = regex::Regex::new(r"\w+").unwrap();
    re.find_iter(input)
        .filter_map(|m| {
            let m_s = m.as_str();

            if m_s.len() > 2 {
                Some(m_s.to_lowercase())
            } else {
                None
            }
        })
        .collect()
}

fn main() {
    tools::init_log();

    let mut doc_list = vec![];
    let mut class_list = vec![];
    let mut full_text = vec![];

    for i in 1..26 {
        let spam_file_name = tools::full_file_path(&format!("Ch04/email/spam/{}.txt", i));
        let ham_file_name = tools::full_file_path(&format!("Ch04/email/ham/{}.txt", i));

        let spam_word = text_parse(&std::fs::read_to_string(spam_file_name).unwrap());

        tracing::trace!("begin ham read: i= {i}");
        let ham_word = text_parse(&std::fs::read_to_string(ham_file_name).unwrap());

        doc_list.push(spam_word.clone());
        full_text.extend(spam_word);
        class_list.push(1);

        doc_list.push(ham_word.clone());
        full_text.extend(ham_word);
        class_list.push(0);
    }

    let vocab_list = alg::bayes::create_vocab_list(doc_list.clone());

    let mut traing_set: Vec<_> = (0..50).collect();
    let mut test_set = vec![];
    for _ in 0..10 {
        let rand_idx = rand::rng().random_range(0..traing_set.len());

        test_set.push(traing_set[rand_idx]);
        traing_set.remove(rand_idx);
    }

    tracing::info!("tracing set: {traing_set:?} test_set: {test_set:?}");

    let mut train_mat = Array2::zeros([0, vocab_list.len()]);
    let mut train_labels = vec![];

    for doc_idx in traing_set {
        let data = alg::bayes::words_set_to_vec(&vocab_list, &doc_list[doc_idx]);
        train_mat.push(Axis(0), data.view()).unwrap();
        train_labels.push(class_list[doc_idx]);
    }

    let train_labels = Array1::from_vec(train_labels);

    let (p0_vect, p1_vect, p_abusive) =
        alg::bayes::train_naive_bayes_0(train_mat.view(), train_labels.view());
    tracing::info!("p0_vect= {p0_vect:?} p1_vect= {p1_vect:?} p_abusive= {p_abusive:?}");

    let mut err_count = 0;

    for doc_idx in &test_set {
        let data = alg::bayes::words_set_to_vec(&vocab_list, &doc_list[*doc_idx]);

        let result = alg::bayes::classify_naive_bayes(
            data.view(),
            p0_vect.view(),
            p1_vect.view(),
            p_abusive,
        );

        if result != class_list[*doc_idx] {
            err_count += 1;
        }
    }

    tracing::info!("the error rate is: {err_count}/{}", test_set.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_parse() {
        let input = "This book is the best book on Python or M.L. I have ever laid eyes upon.";
        let expected = vec![
            "this".to_string(),
            "book".to_string(),
            "the".to_string(),
            "best".to_string(),
            "book".to_string(),
            "python".to_string(),
            "have".to_string(),
            "ever".to_string(),
            "laid".to_string(),
            "eyes".to_string(),
            "upon".to_string(),
        ];
        let result = text_parse(&input);
        assert_eq!(result, expected);
    }
}
