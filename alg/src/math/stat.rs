use ndarray::{Array, Dimension, ShapeBuilder};
use num::Float;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};

pub fn randn<S, D, F: Float>(shape: S) -> Array<F, D>
where
    D: Dimension,
    S: ShapeBuilder<Dim = D>,
    StandardNormal: Distribution<F>,
{
    let mut arr = Array::zeros(shape);

    let mut rng = rand::rng();
    let normal = rand_distr::Normal::new(F::zero(), F::one()).unwrap();

    arr.mapv_inplace(|_a| normal.sample(&mut rng));

    arr
}

pub fn rand<S, D, F: Float>(shape: S) -> Array<F, D>
where
    D: Dimension,
    S: ShapeBuilder<Dim = D>,
    F: SampleUniform,
{
    let mut arr = Array::zeros(shape);

    let mut rng = rand::rng();

    arr.mapv_inplace(|_a| rng.random_range(F::zero()..F::one()));

    arr
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, Array4};

    use super::*;

    #[test]
    fn test_rand() {
        let arr: Array2<f32> = randn((3, 4));
        assert_eq!(arr.shape(), &[3, 4]);
        println!("arr: {arr}");

        let arr: Array4<f32> = randn((10, 1, 28, 28));
        assert_eq!(arr.shape(), &[10, 1, 28, 28]);
        println!("arr: {arr}");
    }
}
