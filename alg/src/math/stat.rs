use ndarray::{Array, Dimension, ShapeBuilder};
use rand_distr::Distribution;

pub fn randn<S, D>(shape: S) -> Array<f32, D>
where
    D: Dimension,
    S: ShapeBuilder<Dim = D>,
{
    let mut arr = Array::zeros(shape);

    let mut rng = rand::rng();
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

    arr.mapv_inplace(|_a| normal.sample(&mut rng));

    arr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rand() {
        let arr = randn((3, 4));
        assert_eq!(arr.shape(), &[3, 4]);
        println!("arr: {arr}");

        let arr = randn((10, 1, 28, 28));
        assert_eq!(arr.shape(), &[10, 1, 28, 28]);
        println!("arr: {arr}");
    }
}
