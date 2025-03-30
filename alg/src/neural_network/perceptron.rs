pub fn and(x1: f64, x2: f64) -> u8 {
    let (w1, w2, theta) = (0.5, 0.5, 0.7);

    let tmp = x1 * w1 + x2 * w2;

    if tmp <= theta { 0 } else { 1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_and() {
        assert_eq!(and(0.0, 0.0), 0);
        assert_eq!(and(0.0, 1.0), 0);
        assert_eq!(and(1.0, 0.0), 0);
        assert_eq!(and(1.0, 1.0), 1);
    }
}
