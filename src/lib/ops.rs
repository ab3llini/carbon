use crate::lib::grad::Operation;
use crate::lib::grad::Scalar;
use crate::lib::tensor::Tensor2D;

use std::ops::{Add, Div, Mul, Sub};


impl Add for &Scalar {
    type Output = Scalar;

    fn add(self, rhs: &Scalar) -> Scalar {
        Scalar::compute(self.clone(), rhs.clone(), Operation::Add)
    }
}

impl Add<f64> for &Scalar {
    type Output = Scalar;

    fn add(self, rhs: f64) -> Scalar {
        Scalar::compute(self.clone(), Scalar::new(rhs), Operation::Add)
    }
}

impl Add<&Scalar> for f64 {
    type Output = Scalar;

    fn add(self, rhs: &Scalar) -> Scalar {
        Scalar::compute(Scalar::new(self), rhs.clone(), Operation::Add)
    }
}

impl Sub for &Scalar {
    type Output = Scalar;

    fn sub(self, rhs: Self) -> Scalar {
        Scalar::compute(self.clone(), rhs.clone(), Operation::Sub)
    }
}

impl Sub<f64> for &Scalar {
    type Output = Scalar;

    fn sub(self, rhs: f64) -> Scalar {
        Scalar::compute(self.clone(), Scalar::new(rhs), Operation::Sub)
    }
}

impl Sub<&Scalar> for f64 {
    type Output = Scalar;

    fn sub(self, rhs: &Scalar) -> Scalar {
        Scalar::compute(Scalar::new(self), rhs.clone(), Operation::Sub)
    }
}

impl Mul for &Scalar {
    type Output = Scalar;

    fn mul(self, rhs: Self) -> Scalar {
        Scalar::compute(self.clone(), rhs.clone(), Operation::Mul)
    }
}

impl Mul<f64> for &Scalar {
    type Output = Scalar;

    fn mul(self, rhs: f64) -> Scalar {
        Scalar::compute(self.clone(), Scalar::new(rhs), Operation::Mul)
    }
}

impl Mul<&Scalar> for f64 {
    type Output = Scalar;

    fn mul(self, rhs: &Scalar) -> Scalar {
        Scalar::compute(Scalar::new(self), rhs.clone(), Operation::Mul)
    }
}

impl Div for &Scalar {
    type Output = Scalar;

    fn div(self, rhs: Self) -> Scalar {
        Scalar::compute(self.clone(), rhs.clone(), Operation::Div)
    }
}

impl Div<f64> for &Scalar {
    type Output = Scalar;

    fn div(self, rhs: f64) -> Scalar {
        Scalar::compute(self.clone(), Scalar::new(rhs), Operation::Div)
    }
}

impl Div<&Scalar> for f64 {
    type Output = Scalar;

    fn div(self, rhs: &Scalar) -> Scalar {
        Scalar::compute(Scalar::new(self), rhs.clone(), Operation::Div)
    }
}

impl Add for &Tensor2D {
    type Output = Tensor2D;

    fn add(self, other: Self) -> Tensor2D {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let mut ans = Tensor2D::zeros(self.rows, self.cols);

        for row in 0..self.rows {
            for col in 0..self.cols {
                ans.data[row][col] = &self.data[row][col] + &other.data[row][col]
            }
        }

        ans
    }
}

impl Add<&Tensor2D> for f64 {
    type Output = Tensor2D;

    fn add(self, other: &Tensor2D) -> Tensor2D {
        let mut ans = Tensor2D::zeros(other.rows, other.cols);

        for row in 0..other.rows {
            for col in 0..other.cols {
                ans.data[row][col] = &Scalar::new(self) + &other.data[row][col]
            }
        }

        ans
    }
}

impl Add<f64> for &Tensor2D {
    type Output = Tensor2D;

    fn add(self, other: f64) -> Tensor2D {
        let mut ans = Tensor2D::zeros(self.rows, self.cols);

        for row in 0..self.rows {
            for col in 0..self.cols {
                ans.data[row][col] = &self.data[row][col] + &Scalar::new(other)
            }
        }

        ans
    }
}

impl Sub for &Tensor2D {
    type Output = Tensor2D;

    fn sub(self, other: Self) -> Tensor2D {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let mut ans = Tensor2D::zeros(self.rows, self.cols);

        for row in 0..self.rows {
            for col in 0..self.cols {
                ans.data[row][col] = &self.data[row][col] - &other.data[row][col]
            }
        }

        ans
    }
}

impl Sub<&Tensor2D> for f64 {
    type Output = Tensor2D;

    fn sub(self, other: &Tensor2D) -> Tensor2D {
        let mut ans = Tensor2D::zeros(other.rows, other.cols);

        for row in 0..other.rows {
            for col in 0..other.cols {
                ans.data[row][col] = &Scalar::new(self) - &other.data[row][col]
            }
        }

        ans
    }
}

impl Sub<f64> for &Tensor2D {
    type Output = Tensor2D;

    fn sub(self, other: f64) -> Tensor2D {
        let mut ans = Tensor2D::zeros(self.rows, self.cols);

        for row in 0..self.rows {
            for col in 0..self.cols {
                ans.data[row][col] = &self.data[row][col] - &Scalar::new(other)
            }
        }

        ans
    }
}


impl Mul for &Tensor2D {
    type Output = Tensor2D;

    fn mul(self, other: Self) -> Tensor2D {
        assert_eq!(self.cols, other.rows);

        let mut ans = Tensor2D::zeros(self.rows, other.cols);
        
        for i in 0..self.rows {
            for j in 0..other.cols {

                let mut sum = Scalar::new(0.0);

                for k in 0..self.cols {
                    sum = &sum + &(&self.data[i][k] * &other.data[k][j]);
                }

                ans.data[i][j] = sum;
            }
        }

        ans
    }
}

impl Mul<&Tensor2D> for f64 {
    type Output = Tensor2D;

    fn mul(self, other: &Tensor2D) -> Tensor2D {
        let mut ans = Tensor2D::zeros(other.rows, other.cols);

        for row in 0..other.rows {
            for col in 0..other.cols {
                ans.data[row][col] = &Scalar::new(self) * &other.data[row][col]
            }
        }

        ans
    }
}

impl Mul<f64> for &Tensor2D {
    type Output = Tensor2D;

    fn mul(self, other: f64) -> Tensor2D {
        let mut ans = Tensor2D::zeros(self.rows, self.cols);

        for row in 0..self.rows {
            for col in 0..self.cols {
                ans.data[row][col] = &self.data[row][col] * &Scalar::new(other)
            }
        }

        ans
    }
}