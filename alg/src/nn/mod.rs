use vectra::prelude::*;

type Ft = f32;
type Mat<T = Ft> = Array<2, T>;
pub fn float_epsilon() -> Ft {
    1.0e-7
}
pub fn matmul_policy() -> MatmulPolicy {
    MatmulPolicy::default()
}

pub mod layer;
pub mod model;
pub mod optimizer;
pub mod train;
