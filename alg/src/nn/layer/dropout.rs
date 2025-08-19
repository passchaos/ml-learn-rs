use crate::{
    math::stat::randn,
    nn::{Float, Mat, layer::LayerWard},
};

pub struct Dropout {
    ratio: Float,
    mask: Option<Mat>,
}

impl Dropout {
    pub fn new(ratio: Float) -> Self {
        Dropout { ratio, mask: None }
    }
}

impl LayerWard for Dropout {
    fn forward(&mut self, input: &Mat) -> Mat {
        let mask =
            randn::<_, _, Float>(input.raw_dim()).map(|x| if x < &self.ratio { 0.0 } else { 1.0 });

        let v = &mask * input;
        self.mask = Some(mask);

        v
    }

    fn backward(&mut self, grad: &Mat) -> Mat {
        self.mask.as_ref().unwrap() * grad
    }
}
