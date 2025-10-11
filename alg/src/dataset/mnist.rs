use std::{io::Cursor, path::Path};

use byteorder::{BigEndian, ReadBytesExt};
use vectra::{NumExt, prelude::*};

pub fn load_images<P: AsRef<Path>>(path: P) -> Array<2, u8> {
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

    Array::from_vec(data, [samples as usize, (rows * cols) as usize])
}

pub fn load_labels<P: AsRef<Path>>(p: P) -> Array<1, u8> {
    let data = std::fs::read(p).unwrap();

    let mut data_r = Cursor::new(data);
    let magic_number = data_r.read_u32::<BigEndian>().unwrap();

    if magic_number != 2049 {
        panic!("wrong magic number found: {magic_number}");
    }

    // let samples = data_r.read_u32::<BigEndian>().unwrap();

    // 剩下的是数据字段，实际长度为 samples
    let data = data_r.into_inner().split_off(8);

    Array::from(data)
}

pub fn load_mnist<T: NumExt>() -> ((Array<2, T>, Array<2, T>), (Array<2, T>, Array<2, T>)) {
    let mnist_dir = std::env::home_dir().unwrap().join("Work/mnist");

    let train_data_path = mnist_dir.join("train-images.idx3-ubyte");
    let train_data =
        load_images(train_data_path).map_into(|v| T::from(v).unwrap() / T::from(255.0).unwrap());

    let label_data_path = mnist_dir.join("train-labels.idx1-ubyte");
    let train_labels = load_labels(label_data_path).one_hot(10);

    let test_data_path = mnist_dir.join("t10k-images.idx3-ubyte");
    let test_data =
        load_images(test_data_path).map_into(|v| T::from(v).unwrap() / T::from(255.0).unwrap());

    let label_data_path = mnist_dir.join("t10k-labels.idx1-ubyte");
    let test_labels = load_labels(label_data_path).one_hot(10);

    ((train_data, train_labels), (test_data, test_labels))
}

#[cfg(test)]
mod tests {
    use crate::dataset::mnist::{load_images, load_labels};

    #[test]
    fn test_mnist_data_parse() {
        let train_data_path = std::env::home_dir()
            .unwrap()
            .join("Work/mnist/train-images.idx3-ubyte");

        let image_data = load_images(train_data_path);
        println!("shape: {:?}", image_data.shape());

        let train_label_path = std::env::home_dir()
            .unwrap()
            .join("Work/mnist/train-labels.idx1-ubyte");
        let image_label = load_labels(train_label_path);
        let one_hot_labels = image_label.one_hot::<i32>(10);
        println!("one_hot labels: {one_hot_labels:?}");
    }
}
