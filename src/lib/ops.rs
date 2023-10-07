use crate::lib::grad::Dependency;
use crate::lib::grad::Node;
use crate::lib::grad::Operation;
use crate::lib::grad::Scalar;
use crate::lib::tensor::Tensor2D;

use std::cell::Cell;
use std::rc::Rc;

use std::ops::{Add, Div, Mul, Sub};

fn op(lhs: &Scalar, rhs: &Scalar, op: Operation) -> Scalar {
    Scalar {
        node: Rc::new(Node {
            val: Cell::new(match op {
                Operation::Add => lhs.value() + rhs.value(),
                Operation::Sub => lhs.value() - rhs.value(),
                Operation::Mul => lhs.value() * rhs.value(),
                Operation::Div => lhs.value() / rhs.value(),
            }),
            grad: Cell::new(None),
            dep: Some(Dependency::Double {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                op,
            }),
        }),
    }
}

impl Add for &Scalar {
    type Output = Scalar;

    fn add(self, rhs: Self) -> Self::Output {
        op(self, rhs, Operation::Add)
    }
}

impl Add<f64> for &Scalar {
    type Output = Scalar;

    fn add(self, rhs: f64) -> Self::Output {
        op(self, &Scalar::new(rhs), Operation::Add)
    }
}

impl Add<&Scalar> for f64 {
    type Output = Scalar;

    fn add(self, rhs: &Scalar) -> Self::Output {
        op(&Scalar::new(self), rhs, Operation::Add)
    }
}

impl Sub for &Scalar {
    type Output = Scalar;

    fn sub(self, rhs: Self) -> Self::Output {
        op(self, rhs, Operation::Sub)
    }
}

impl Sub<f64> for &Scalar {
    type Output = Scalar;

    fn sub(self, rhs: f64) -> Self::Output {
        op(self, &Scalar::new(rhs), Operation::Sub)
    }
}

impl Sub<&Scalar> for f64 {
    type Output = Scalar;

    fn sub(self, rhs: &Scalar) -> Self::Output {
        op(&Scalar::new(self), rhs, Operation::Sub)
    }
}

impl Mul for &Scalar {
    type Output = Scalar;

    fn mul(self, rhs: Self) -> Self::Output {
        op(self, rhs, Operation::Mul)
    }
}

impl Mul<f64> for &Scalar {
    type Output = Scalar;

    fn mul(self, rhs: f64) -> Self::Output {
        op(self, &Scalar::new(rhs), Operation::Mul)
    }
}

impl Mul<&Scalar> for f64 {
    type Output = Scalar;

    fn mul(self, rhs: &Scalar) -> Self::Output {
        op(&Scalar::new(self), rhs, Operation::Mul)
    }
}

impl Div for &Scalar {
    type Output = Scalar;

    fn div(self, rhs: Self) -> Self::Output {
        op(self, rhs, Operation::Div)
    }
}

impl Div<f64> for &Scalar {
    type Output = Scalar;

    fn div(self, rhs: f64) -> Self::Output {
        op(self, &Scalar::new(rhs), Operation::Div)
    }
}

impl Div<&Scalar> for f64 {
    type Output = Scalar;

    fn div(self, rhs: &Scalar) -> Self::Output {
        op(&Scalar::new(self), rhs, Operation::Div)
    }
}

impl Add for &Tensor2D {
    type Output = Tensor2D;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let mut ans = Tensor2D::zeros(self.rows, self.cols);

        for row in 0..self.rows {
            for col in 0..self.cols {
                ans.data[row][col] = op(&self.data[row][col], &rhs.data[row][col], Operation::Add);
            }
        }

        ans
    }
}

impl Add<f64> for &Tensor2D {
    type Output = Tensor2D;

    fn add(self, rhs: f64) -> Self::Output {
        let mut ans = Tensor2D::zeros(self.rows, self.cols);

        for row in 0..self.rows {
            for col in 0..self.cols {
                ans.data[row][col] = op(&self.data[row][col], &Scalar::new(rhs), Operation::Add);
            }
        }

        ans
    }
}

impl Add<&Tensor2D> for f64 {
    type Output = Tensor2D;

    fn add(self, rhs: &Tensor2D) -> Self::Output {
        let mut ans = Tensor2D::zeros(rhs.rows, rhs.cols);

        for row in 0..rhs.rows {
            for col in 0..rhs.cols {
                ans.data[row][col] = op(&Scalar::new(self), &rhs.data[row][col], Operation::Add);
            }
        }

        ans
    }
}

impl Sub for &Tensor2D {
    type Output = Tensor2D;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let mut ans = Tensor2D::zeros(self.rows, self.cols);

        for row in 0..self.rows {
            for col in 0..self.cols {
                ans.data[row][col] = op(&self.data[row][col], &rhs.data[row][col], Operation::Sub);
            }
        }

        ans
    }
}

impl Sub<f64> for &Tensor2D {
    type Output = Tensor2D;

    fn sub(self, rhs: f64) -> Self::Output {
        let mut ans = Tensor2D::zeros(self.rows, self.cols);

        for row in 0..self.rows {
            for col in 0..self.cols {
                ans.data[row][col] = op(&self.data[row][col], &Scalar::new(rhs), Operation::Sub);
            }
        }

        ans
    }
}

impl Sub<&Tensor2D> for f64 {
    type Output = Tensor2D;

    fn sub(self, rhs: &Tensor2D) -> Self::Output {
        let mut ans = Tensor2D::zeros(rhs.rows, rhs.cols);

        for row in 0..rhs.rows {
            for col in 0..rhs.cols {
                ans.data[row][col] = op(&Scalar::new(self), &rhs.data[row][col], Operation::Sub);
            }
        }

        ans
    }
}

impl Mul for &Tensor2D {
    type Output = Tensor2D;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.cols, rhs.rows,
            "{}x{} incompatible with {}x{}",
            self.rows, self.cols, rhs.rows, rhs.cols
        );

        let mut ans = Tensor2D::zeros(self.rows, rhs.cols);

        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut sum = Scalar::new(0.0);

                for k in 0..self.cols {
                    sum = op(
                        &sum,
                        &op(&self.data[i][k], &rhs.data[k][j], Operation::Mul),
                        Operation::Add,
                    );
                }

                ans.data[i][j] = sum;
            }
        }

        ans
    }
}

impl Mul<f64> for &Tensor2D {
    type Output = Tensor2D;

    fn mul(self, rhs: f64) -> Self::Output {
        let mut ans = Tensor2D::zeros(self.rows, self.cols);

        for row in 0..self.rows {
            for col in 0..self.cols {
                ans.data[row][col] = op(&self.data[row][col], &Scalar::new(rhs), Operation::Mul);
            }
        }

        ans
    }
}

impl Mul<&Tensor2D> for f64 {
    type Output = Tensor2D;

    fn mul(self, rhs: &Tensor2D) -> Self::Output {
        let mut ans = Tensor2D::zeros(rhs.rows, rhs.cols);

        for row in 0..rhs.rows {
            for col in 0..rhs.cols {
                ans.data[row][col] = op(&Scalar::new(self), &rhs.data[row][col], Operation::Mul);
            }
        }

        ans
    }
}
