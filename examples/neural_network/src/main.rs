extern crate openblas_src;

use std::collections::HashMap;

use alg::math::Sigmoid;
use ndarray::{Array2, array};

fn init_network() -> HashMap<String, Array2<f32>> {
    let mut network = HashMap::new();

    network.insert("W1".to_string(), array![[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]);
    network.insert("b1".to_string(), array![[0.1, 0.2, 0.3]]);
    network.insert("W2".to_string(), array![[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]);
    network.insert("b2".to_string(), array![[0.1, 0.2]]);
    network.insert("W3".to_string(), array![[0.1, 0.3], [0.2, 0.4]]);
    network.insert("b3".to_string(), array![[0.1, 0.2]]);

    network
}

fn forward(mut network: HashMap<String, Array2<f32>>, x: Array2<f32>) -> Array2<f32> {
    let W1 = network.remove("W1").unwrap();
    let W2 = network.remove("W2").unwrap();
    let W3 = network.remove("W3").unwrap();

    let b1 = network.remove("b1").unwrap();
    let b2 = network.remove("b2").unwrap();
    let b3 = network.remove("b3").unwrap();

    let a1 = x.dot(&W1) + &b1;
    let z1 = a1.sigmoid();

    let a2 = z1.dot(&W2) + &b2;
    let z2 = a2.sigmoid();

    let a3 = z2.dot(&W3) + &b3;
    a3
}
fn main() {
    let network = init_network();
    let x = array![[1.0, 0.5]];
    let y = forward(network, x);
    println!("{y}");
}
