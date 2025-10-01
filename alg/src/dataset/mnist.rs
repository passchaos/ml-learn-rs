use std::{io::Cursor, path::Path};

use byteorder::{BigEndian, ReadBytesExt};
use ndarray::{Array1, Array2};

pub fn load_images<P: AsRef<Path>>(path: P) -> Array2<u8> {
    let data = std::fs::read(path).unwrap();

    let mut data_r = Cursor::new(data);

    let magic_number = data_r.read_u32::<BigEndian>().unwrap();

    if magic_number != 2051 {
        panic!("wrong magic number found: {magic_number}");
    }

    let samples = data_r.read_u32::<BigEndian>().unwrap();
    let rows = data_r.read_u32::<BigEndian>().unwrap();
    let cols = data_r.read_u32::<BigEndian>().unwrap();

    // 剩下的是数据字段，实际长度为 samples * rows * cols
    let data = data_r.into_inner().split_off(16);
    println!("data len: {} length= {}", data.len(), samples * rows * cols);

    Array2::from_shape_vec((samples as usize, (rows * cols) as usize), data).unwrap()
}

pub fn load_labels<P: AsRef<Path>>(p: P) -> Array1<u8> {
    let data = std::fs::read(p).unwrap();

    let mut data_r = Cursor::new(data);
    let magic_number = data_r.read_u32::<BigEndian>().unwrap();

    if magic_number != 2049 {
        panic!("wrong magic number found: {magic_number}");
    }

    // let samples = data_r.read_u32::<BigEndian>().unwrap();

    // 剩下的是数据字段，实际长度为 samples
    let data = data_r.into_inner().split_off(8);

    Array1::from_vec(data)
}

#[cfg(test)]
mod tests {
    use crate::{
        dataset::mnist::{load_images, load_labels},
        math::{DigitalRecognition, one_hot::OneHotTransform},
    };

    #[test]
    fn test_mnist_data_parse() {
        let train_data_path = std::env::home_dir()
            .unwrap()
            .join("Work/mnist/train-images.idx3-ubyte");

        let image_data = load_images(train_data_path);

        let train_label_path = std::env::home_dir()
            .unwrap()
            .join("Work/mnist/train-labels.idx1-ubyte");
        let image_label = load_labels(train_label_path);

        let oneshoted_labels = DigitalRecognition::one_hot(&image_label);
        println!(
            "image_data= {image_data:?} image_label= {image_label:?} oneshoted_labels= {oneshoted_labels:?}"
        );
    }
}
