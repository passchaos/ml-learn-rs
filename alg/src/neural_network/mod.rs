pub mod perceptron;

#[cfg(test)]
mod tests {
    use ndarray::array;

    #[test]
    fn test_misc() {
        let a = array![1, 2, 3];
        let b = a.dot(&a);
        println!("a dot a: {b}");

        let a1 = array![[1], [2], [3]];
        let b = array![[1, 2, 3], [4, 5, 6]];
        let c = b.dot(&a);
        let c1 = b.dot(&a1);

        println!("output: c= {c} c1= {c1}");
    }
}
