pub mod bayes;
pub mod dataset;
pub mod math;
pub mod nlp;
pub mod nn;
pub mod tensor;
pub mod util;

#[cfg(test)]
mod tests {
    use ndarray::*;
    use ndarray_linalg::*;

    #[test]
    fn test_svd() {
        let a = arr2(&[
            [3.0, 4.0, 3.0],
            [1.0, 2.0, 3.0],
            [4.0, 2.0, 1.0],
            [1.0, 1.0, 1.0],
        ]);

        let (u, s, vt) = a.svd(true, true).unwrap();

        let u = u.unwrap();
        let vt = vt.unwrap();

        println!("U: {u} S: {s} Vt: {vt}");

        let (m, n) = a.dim();

        let mut sm = Array2::zeros((m, n));

        for i in 0..std::cmp::min(m, n) {
            sm[(i, i)] = s[i];
        }

        println!("sm: {sm}");

        let res = u.dot(&sm).dot(&vt);
        println!("res: {res}");
    }
}
