use ndarray::{Array2, Array4};

pub fn im2col<T>(
    input_data: Array4<T>,
    filter_height: usize,
    filter_width: usize,
    stridge: usize,
    pad: usize,
) -> Array2<T> {
    let input_shape = input_data.shape();

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let height = input_shape[2];
    let width = input_shape[3];

    let out_h = ((height + 2 * pad - filter_height) / stridge) + 1;
    let out_w = ((width + 2 * pad - filter_width) / stridge) + 1;

    todo!()
}

pub fn col2im<T>(
    col_data: Array2<T>,
    input_shape: (usize, usize, usize, usize),
    filter_height: usize,
    filter_width: usize,
    stridge: usize,
    pad: usize,
) -> Array4<T> {
    todo!()
}
