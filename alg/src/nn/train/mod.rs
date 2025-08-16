use ndarray::{ArrayBase, RawData};

pub trait Model<SL: RawData, DL, SR: RawData, DR> {
    fn forward(&mut self, input: &ArrayBase<SL, DL>, label: &ArrayBase<SR, DR>);
    fn backward(&mut self);
}
pub struct Trainer {}
