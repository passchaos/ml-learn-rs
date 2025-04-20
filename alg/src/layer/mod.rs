pub mod add;
pub mod affine;
pub mod mul;
pub mod relu;
pub mod sigmoid;
pub mod softmax_loss;

#[cfg(test)]
mod tests {
    use ndarray::{Array1, ArrayView, array};

    use crate::math::{Relu, autodiff::numerical_gradient};

    use super::{add::AddLayer, mul::MulLayer, relu::ReluLayer};

    #[test]
    fn test_add_mul_layer() {
        let apple = 100.0;
        let apple_num = 2.0;
        let orange = 150.0;
        let orange_num = 3.0;
        let tax = 1.1;

        let mut mul_apple_layer = MulLayer::default();
        let mut mul_orange_layer = MulLayer::default();
        let mut add_apple_orange_layer = AddLayer::default();
        let mut mul_tax_layer = MulLayer::default();

        let apple_price = mul_apple_layer.forward(apple, apple_num);
        let orange_price = mul_orange_layer.forward(orange, orange_num);
        let all_price = add_apple_orange_layer.forward(apple_price, orange_price);
        let price = mul_tax_layer.forward(all_price, tax);

        let dprice = 1.0;
        let (dall_price, dtax) = mul_tax_layer.backward(dprice);
        let (dapple_price, dorange_price) = add_apple_orange_layer.backward(dall_price);
        let (dapple, dapple_num) = mul_apple_layer.backward(dapple_price);
        let (dorange, dorange_num) = mul_orange_layer.backward(dorange_price);

        println!("price: {price}");
        println!(
            "dapple_num= {dapple_num} dapple= {dapple} dorange_num= {dorange_num} dorange= {dorange} dtax= {dtax}"
        );
    }

    #[test]
    fn test_relu() {
        let mut layer = ReluLayer::default();
        let mut x = array![-4, 1, 2, 3];
        let res = layer.forward(&x.view());

        println!("layer: {layer:?}");
        assert_eq!(res, array![0, 1, 2, 3]);

        let d_x = layer.backward(&x.view());

        assert_eq!(d_x, array![0, 1, 2, 3]);
        println!("res: {res} d_x= {d_x}");
    }
}
