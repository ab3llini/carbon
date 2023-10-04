// Wrapper for Scalar::new
macro_rules! scalar {
    ($value:expr) => {
        Scalar::new($value)
    };
}

// Wrapper for Tensor2D::from_vec
macro_rules! tensor {
    ($vec:expr) => {
        Tensor2D::from($vec)
    };
}

pub(crate) use scalar;
pub(crate) use tensor;
