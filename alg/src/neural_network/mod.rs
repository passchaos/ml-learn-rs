pub mod perceptron;

#[cfg(test)]
mod tests {
    use crate::math::Sigmoid;
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

        let x = array![1.0, 2.0];
        let x1 = array![[1.0, 2.0]];
        let y = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let dot1 = x.dot(&y);
        let dot2 = x1.dot(&y);
        println!(
            "output 1: {} s= {} {} s= {}",
            dot1,
            dot1.sigmoid(),
            dot2,
            dot2.sigmoid()
        );
    }
}
