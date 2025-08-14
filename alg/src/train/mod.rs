use ndarray::ArrayViewD;

pub trait Model<F> {
    pub fn forward(&mut self, input: &ArrayViewD<F>, )
}
pub struct Trainer {}
