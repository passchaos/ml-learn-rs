mod knn;

struct Data {
    a: f64,
    b: f64,
}

impl From<(f64, f64)> for Data {
    fn from(value: (f64, f64)) -> Self {
        Self {
            a: value.0,
            b: value.1,
        }
    }
}

impl Data {
    fn distance(&self, other: &Data) -> f64 {
        ((self.a - other.a).powi(2) + (self.b - other.b).powi(2)).sqrt()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum Label {
    A,
    B,
}

struct SampleData {
    data: Data,
    label: Label,
}

fn main() {
    let sample_datas: Vec<_> = vec![
        ((1.0, 1.1), Label::A),
        ((1.0, 1.0), Label::A),
        ((0.0, 0.0), Label::B),
        ((0.0, 0.1), Label::B),
    ]
    .into_iter()
    .map(|(data, label)| SampleData {
        data: data.into(),
        label,
    })
    .collect();

    let res = knn::classify(Data { a: 0.0, b: 0.0 }, &sample_datas, 3);
    println!("result: {res:?}");
}
