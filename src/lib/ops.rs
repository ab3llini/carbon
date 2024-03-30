use crate::lib::grad::Data;
use crate::lib::grad::Dependency;
use crate::lib::grad::Scalar;
use crate::lib::tensor::Tensor2D;

use std::cell::RefCell;
use std::ops::{ Add, Div, Mul, Sub };
use std::rc::Rc;

#[derive(Debug, Clone)]
pub enum Operation {
    Add,
    Mul,
    Sub,
    Div,
}

fn op(lhs: &Scalar, rhs: &Scalar, op: Operation) -> Scalar {
    // let requires_grad: bool = lhs.data.borrow().requires_grad || rhs.data.borrow().requires_grad;
    let requires_grad: bool = true;

    Scalar {
        data: Rc::new(
            RefCell::new(Data {
                val: match op {
                    Operation::Add => lhs.val() + rhs.val(),
                    Operation::Sub => lhs.val() - rhs.val(),
                    Operation::Mul => lhs.val() * rhs.val(),
                    Operation::Div => lhs.val() / rhs.val(),
                },
                grad: 0.0,
                dep: match requires_grad {
                    true =>
                        Some(Dependency::Double {
                            lhs: Rc::clone(&lhs.data),
                            rhs: Rc::clone(&rhs.data),
                            op,
                        }),
                    false => None,
                },
                requires_grad,
            })
        ),
    }
}

impl Add for &Scalar {
    type Output = Scalar;

    fn add(self, rhs: Self) -> Self::Output {
        op(self, rhs, Operation::Add)
    }
}

impl Add<f32> for &Scalar {
    type Output = Scalar;

    fn add(self, rhs: f32) -> Self::Output {
        op(self, &Scalar::new(rhs, false), Operation::Add)
    }
}

impl Add<&Scalar> for f32 {
    type Output = Scalar;

    fn add(self, rhs: &Scalar) -> Self::Output {
        op(&Scalar::new(self, false), rhs, Operation::Add)
    }
}

impl Sub for &Scalar {
    type Output = Scalar;

    fn sub(self, rhs: Self) -> Self::Output {
        op(self, rhs, Operation::Sub)
    }
}

impl Sub<f32> for &Scalar {
    type Output = Scalar;

    fn sub(self, rhs: f32) -> Self::Output {
        op(self, &Scalar::new(rhs, false), Operation::Sub)
    }
}

impl Sub<&Scalar> for f32 {
    type Output = Scalar;

    fn sub(self, rhs: &Scalar) -> Self::Output {
        op(&Scalar::new(self, false), rhs, Operation::Sub)
    }
}

impl Mul for &Scalar {
    type Output = Scalar;

    fn mul(self, rhs: Self) -> Self::Output {
        op(self, rhs, Operation::Mul)
    }
}

impl Mul<f32> for &Scalar {
    type Output = Scalar;

    fn mul(self, rhs: f32) -> Self::Output {
        op(self, &Scalar::new(rhs, false), Operation::Mul)
    }
}

impl Mul<&Scalar> for f32 {
    type Output = Scalar;

    fn mul(self, rhs: &Scalar) -> Self::Output {
        op(&Scalar::new(self, false), rhs, Operation::Mul)
    }
}

impl Div for &Scalar {
    type Output = Scalar;

    fn div(self, rhs: Self) -> Self::Output {
        op(self, rhs, Operation::Div)
    }
}

impl Div<f32> for &Scalar {
    type Output = Scalar;

    fn div(self, rhs: f32) -> Self::Output {
        op(self, &Scalar::new(rhs, false), Operation::Div)
    }
}

impl Div<&Scalar> for f32 {
    type Output = Scalar;

    fn div(self, rhs: &Scalar) -> Self::Output {
        op(&Scalar::new(self, false), rhs, Operation::Div)
    }
}

impl Add for &Tensor2D {
    type Output = Tensor2D;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let mut ans = Tensor2D::zeros(self.rows, self.cols, false);

        for row in 0..self.rows {
            for col in 0..self.cols {
                ans.data[row][col] = op(&self.data[row][col], &rhs.data[row][col], Operation::Add);
            }
        }

        ans
    }
}

impl Add<f32> for &Tensor2D {
    type Output = Tensor2D;

    fn add(self, rhs: f32) -> Self::Output {
        let mut ans = Tensor2D::zeros(self.rows, self.cols, false);

        for row in 0..self.rows {
            for col in 0..self.cols {
                ans.data[row][col] = op(
                    &self.data[row][col],
                    &Scalar::new(rhs, false),
                    Operation::Add
                );
            }
        }

        ans
    }
}

impl Add<&Tensor2D> for f32 {
    type Output = Tensor2D;

    fn add(self, rhs: &Tensor2D) -> Self::Output {
        let mut ans = Tensor2D::zeros(rhs.rows, rhs.cols, false);

        for row in 0..rhs.rows {
            for col in 0..rhs.cols {
                ans.data[row][col] = op(
                    &Scalar::new(self, false),
                    &rhs.data[row][col],
                    Operation::Add
                );
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

        let mut ans = Tensor2D::zeros(self.rows, self.cols, false);

        for row in 0..self.rows {
            for col in 0..self.cols {
                ans.data[row][col] = op(&self.data[row][col], &rhs.data[row][col], Operation::Sub);
            }
        }

        ans
    }
}

impl Sub<f32> for &Tensor2D {
    type Output = Tensor2D;

    fn sub(self, rhs: f32) -> Self::Output {
        let mut ans = Tensor2D::zeros(self.rows, self.cols, false);

        for row in 0..self.rows {
            for col in 0..self.cols {
                ans.data[row][col] = op(
                    &self.data[row][col],
                    &Scalar::new(rhs, false),
                    Operation::Sub
                );
            }
        }

        ans
    }
}

impl Sub<&Tensor2D> for f32 {
    type Output = Tensor2D;

    fn sub(self, rhs: &Tensor2D) -> Self::Output {
        let mut ans = Tensor2D::zeros(rhs.rows, rhs.cols, false);

        for row in 0..rhs.rows {
            for col in 0..rhs.cols {
                ans.data[row][col] = op(
                    &Scalar::new(self, false),
                    &rhs.data[row][col],
                    Operation::Sub
                );
            }
        }

        ans
    }
}

impl Mul for &Tensor2D {
    type Output = Tensor2D;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.cols,
            rhs.rows,
            "{}x{} incompatible with {}x{}",
            self.rows,
            self.cols,
            rhs.rows,
            rhs.cols
        );

        let mut ans = Tensor2D::zeros(self.rows, rhs.cols, false);

        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut sum: Option<Scalar> = None;

                for k in 0..self.cols {
                    match &sum {
                        None => {
                            sum = Some(op(&self.data[i][k], &rhs.data[k][j], Operation::Mul));
                        }
                        Some(_) => {
                            sum = Some(
                                op(
                                    &sum.unwrap(),
                                    &op(&self.data[i][k], &rhs.data[k][j], Operation::Mul),
                                    Operation::Add
                                )
                            );
                        }
                    }
                }

                ans.data[i][j] = sum.unwrap();
            }
        }
        ans
    }
}

impl Mul<f32> for &Tensor2D {
    type Output = Tensor2D;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut ans = Tensor2D::zeros(self.rows, self.cols, false);

        for row in 0..self.rows {
            for col in 0..self.cols {
                ans.data[row][col] = op(
                    &self.data[row][col],
                    &Scalar::new(rhs, false),
                    Operation::Mul
                );
            }
        }

        ans
    }
}

impl Mul<&Tensor2D> for f32 {
    type Output = Tensor2D;

    fn mul(self, rhs: &Tensor2D) -> Self::Output {
        let mut ans = Tensor2D::zeros(rhs.rows, rhs.cols, false);

        for row in 0..rhs.rows {
            for col in 0..rhs.cols {
                ans.data[row][col] = op(
                    &Scalar::new(self, false),
                    &rhs.data[row][col],
                    Operation::Mul
                );
            }
        }

        ans
    }
}
