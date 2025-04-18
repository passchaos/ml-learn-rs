#[derive(Debug, Default)]
pub struct MulLayer {
    x: Option<f32>,
    y: Option<f32>,
}

impl MulLayer {
    pub fn forward(&mut self, x: f32, y: f32) -> f32 {
        self.x = Some(x);
        self.y = Some(y);

        let out = x * y;
        out
    }

    pub fn backward(&self, dout: f32) -> (f32, f32) {
        let dx = dout * self.y.unwrap();
        let dy = dout * self.x.unwrap();

        (dx, dy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward() {
        let mut layer = MulLayer::default();
        let x = 2.0;
        let y = 3.0;
        let out = layer.forward(x, y);

        let mut z_layer = MulLayer::default();
        let z = -2.7;
        let out_z = z_layer.forward(out, z);
        assert_eq!(out, 6.0);
        assert_eq!(out_z, -16.2);

        let (y_z_d, z_d) = z_layer.backward(1.0);
        let (x_d, y_d) = layer.backward(y_z_d);

        assert_eq!(z_d, 6.0);
        assert_eq!(y_z_d, -2.7);
        assert_eq!(y_d, -2.7 * x);
        assert_eq!(x_d, -2.7 * y);
    }
}
