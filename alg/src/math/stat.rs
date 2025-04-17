use ndarray::{Array, Dimension, ShapeBuilder};
use rand::Rng;

pub fn randn<S, D>(shape: S) -> Array<f32, D>
where
    D: Dimension,
    S: ShapeBuilder<Dim = D>,
{
    let mut arr = Array::zeros(shape);

    let mut rng = rand::rng();

    arr.mapv_inplace(|_a| rng.random_range(0.0..1.0));

    arr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rand() {
        let arr = randn((3, 4));
        assert_eq!(arr.shape(), &[3, 4]);
        assert!(arr.iter().all(|&x| x >= 0.0 && x < 1.0));
        println!("arr: {arr}");
    }
}
