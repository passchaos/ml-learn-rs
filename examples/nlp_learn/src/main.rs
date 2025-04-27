use std::collections::HashMap;

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
fn main() {
    let text = "You say goodbye and I say hello.";
    let res = preprocess(text);
    println!("res: {res:?}");
}
