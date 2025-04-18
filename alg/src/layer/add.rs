#[derive(Debug, Default)]
pub struct AddLayer;

impl AddLayer {
    pub fn forward(&self, x: f32, y: f32) -> f32 {
        return x + y;
    }

    pub fn backward(&self, dout: f32) -> (f32, f32) {
        (dout, dout)
    }
}
