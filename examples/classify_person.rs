extern crate ml_learn_rs;

use std::io::Write;

use ndarray::array;

use ml_learn_rs::{knn, tools};

fn main() {
    tools::init_logs();

    let result_list = ["not at all", "in small doses", "in large doses"];

    let mut input = String::new();

    print!("percentage of time spent playing video game? ");
    std::io::stdout().flush().unwrap();

    std::io::stdin().read_line(&mut input).unwrap();
    let percent_tats: f64 = input.trim().parse().unwrap();

    input = String::new();
    print!("frequent flier miles earned per year? ");
    std::io::stdout().flush().unwrap();

    std::io::stdin().read_line(&mut input).unwrap();
    let ff_miles: f64 = input.trim().parse().unwrap();

    input = String::new();
    print!("liters of ice cream consumed per year? ");
    std::io::stdout().flush().unwrap();
    std::io::stdin().read_line(&mut input).unwrap();
    let ice_cream: f64 = input.trim().parse().unwrap();

    let (dating_data_mat, dating_labels) =
        knn::file2matrix(crate::tools::full_file_path("Ch02/datingTestSet2.txt"));

    let (norm_mat, ranges, min_vals) = knn::auto_norm(dating_data_mat.view());

    let in_arr = (array![ff_miles, percent_tats, ice_cream] - min_vals) / ranges;

    let classifier_result = knn::classify(in_arr.view(), norm_mat.view(), dating_labels.view(), 3);

    let v: usize = classifier_result.parse().unwrap();
    println!("You will probably like this person: {}", result_list[v - 1]);
}
