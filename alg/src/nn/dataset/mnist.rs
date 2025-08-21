use std::{io::Cursor, path::Path};

use byteorder::{BigEndian, ReadBytesExt};

use crate::nn::{Float, Tensor1, Tensor2, default_device};

pub fn load_images<P: AsRef<Path>>(path: P) -> Tensor2 {
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
    let data: Vec<_> = data_r
        .into_inner()
        .split_off(16)
        .into_iter()
        .map(|a| a as Float / 255.0)
        .collect();

    println!("data len: {} length= {}", data.len(), samples * rows * cols);

    Tensor1::from_data(data.as_slice(), &default_device())
        .reshape([samples as usize, (rows * cols) as usize])
}

pub fn load_labels<P: AsRef<Path>>(p: P) -> Tensor2 {
    let data = std::fs::read(p).unwrap();

    let mut data_r = Cursor::new(data);
    let magic_number = data_r.read_u32::<BigEndian>().unwrap();

    if magic_number != 2049 {
        panic!("wrong magic number found: {magic_number}");
    }

    // let samples = data_r.read_u32::<BigEndian>().unwrap();

    // 剩下的是数据字段，实际长度为 samples
    let data: Vec<_> = data_r
        .into_inner()
        .split_off(8)
        .into_iter()
        .map(|a| a as Float)
        .collect();

    let t = Tensor1::from_data(data.as_slice(), &default_device());
    t.one_hot(10)
}
