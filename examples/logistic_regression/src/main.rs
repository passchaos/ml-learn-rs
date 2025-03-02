extern crate openblas_src;

use std::io::BufRead;

use alg::math::sigmoid;
use ndarray::Array2;

fn load_data_set() -> (Array2<f64>, Array2<f64>) {
    let path = tools::full_file_path("Ch05/testSet.txt");

    let file = std::io::BufReader::new(std::fs::File::open(path).unwrap());

    let mut data_mat = vec![];
    let mut label_vec = vec![];

    let mut m = 0;
    for line in file.lines() {
        let line = line.unwrap();

        let mut a = line.trim().split_whitespace();

        // println!("a: {a:?}");

        let line_0 = a.next().unwrap().parse().unwrap();
        let line_1 = a.next().unwrap().parse().unwrap();
        let line_2: f64 = a.next().unwrap().parse().unwrap();

        data_mat.extend([1.0, line_0, line_1]);
        label_vec.push(line_2);
        m += 1;
    }

    (
        Array2::from_shape_vec((m, 3), data_mat).unwrap(),
        Array2::from_shape_vec((m, 1), label_vec).unwrap(),
    )
}

fn gradient_ascent(data_in: Array2<f64>, labels_in: Array2<f64>) -> Array2<f64> {
    let n = data_in.shape()[1];
    let alpha = 0.001;
    let max_cycles = 500;

    let mut weights: Array2<f64> = Array2::ones((n, 1));

    for _ in 0..max_cycles {
        let h = sigmoid(data_in.dot(&weights));

        let error = &labels_in - &h;
        weights = weights + alpha * data_in.t().dot(&error);
    }

    weights
}

fn main() {
    let (data_in, labels_in) = load_data_set();

    let weights = gradient_ascent(data_in, labels_in);

    println!("weights: {:?}", weights);
}
