use ndarray::{Array, Array1, Array2, ArrayView, Axis, Dimension, LinalgScalar, linalg::Dot};

pub struct AffineLayer<T> {
    w: Array2<T>,
    b: Array1<T>,
    x: Option<Array2<T>>,
}

impl<T> AffineLayer<T> {
    pub fn new(w: Array2<T>, b: Array1<T>) -> Self {
        Self { w, b, x: None }
    }
}

impl<T: Clone + LinalgScalar> AffineLayer<T> {
    pub fn forward(&mut self, x: &Array2<T>) -> Array2<T> {
        self.x = Some(x.clone());

        let out = x.dot(&self.w) + &self.b;

        out
    }

    pub fn backward(&self, dout: &Array2<T>) -> (Array2<T>, Array2<T>, Array1<T>) {
        let dx = dout.dot(&self.w.t());

        let dw = self.x.as_ref().unwrap().t().dot(dout);
        let db = dout.sum_axis(Axis(0));

        (dx, dw, db)
    }
}
