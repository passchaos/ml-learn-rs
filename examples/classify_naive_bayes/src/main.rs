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
    let input = "This book is the best book on Python or M.L. I have ever laid eyes upon.";

    let result = text_parse(&input);
    println!("{result:?}");
}
