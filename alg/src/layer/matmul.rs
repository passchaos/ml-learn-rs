use ndarray::{Array2, ArrayView2, LinalgScalar};

pub struct MatMulLayer<T> {
    w: Array2<T>,
    x: Option<Array2<T>>,
}

impl<T> MatMulLayer<T> {
    pub fn new(w: Array2<T>) -> Self {
        Self { w, x: None }
    }
}

impl<T: LinalgScalar> MatMulLayer<T> {
    pub fn forward(&mut self, x: &ArrayView2<T>) -> Array2<T> {
        self.x = Some(x.to_owned());

        let out = x.dot(&self.w);

        out
    }

    pub fn backward(&self, dout: &ArrayView2<T>) -> (Array2<T>, Array2<T>) {
        let dx = dout.dot(&self.w.t());

        let dw = self.x.as_ref().unwrap().t().dot(dout);
        (dx, dw)
    }
}
