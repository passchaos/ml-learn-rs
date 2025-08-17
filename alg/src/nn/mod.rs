type Float = f32;
type Mat<T = Float> = ndarray::Array2<T>;
type Mat1<T = Float> = ndarray::Array1<T>;
pub fn float_epsilon() -> Float {
    1.0e-7
}

pub mod layer;
pub mod optimizer;
pub mod train;
