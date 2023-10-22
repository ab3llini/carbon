use crate::lib::grad::Activation;
use crate::lib::grad::Nonlinear;
use crate::lib::grad::Scalar;

use rand::distributions::Uniform;
use rand::Rng;
use std::cell::RefCell;
use std::rc::Rc;
use std::vec::Vec;

use super::grad::Data;

#[derive(Debug, Clone)]
pub struct Tensor2D {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<Scalar>>,
}

impl Tensor2D {
    pub fn zeros(rows: usize, cols: usize, requires_grad: bool) -> Self {
        let data = {
            let mut data = Vec::new();
            for _ in 0..rows {
                let mut row = Vec::new();
                for _ in 0..cols {
                    row.push(Scalar::new(0.0, requires_grad));
                }
                data.push(row);
            }
            data
        };

        Self { rows, cols, data }
    }

    pub fn uniform(rows: usize, cols: usize, requires_grad: bool) -> Self {
        let zeros = Self::zeros(rows, cols, requires_grad);
        let mut rng = rand::thread_rng();
        let side = Uniform::new(-1.0, 1.0);

        for row in 0..zeros.rows {
            for col in 0..zeros.cols {
                zeros.data[row][col].data.borrow_mut().val = rng.sample(side)
            }
        }

        Self {
            rows,
            cols,
            data: zeros.data,
        }
    }

    pub fn xavier(rows: usize, cols: usize, requires_grad: bool) -> Self {
        let zeros = Self::zeros(rows, cols, requires_grad);
        let mut rng = rand::thread_rng();
        let side = Uniform::new(-1.0, 1.0);

        for row in 0..zeros.rows {
            for col in 0..zeros.cols {
                zeros.data[row][col].data.borrow_mut().val =
                    rng.sample(side) / (rows as f32).sqrt();
            }
        }

        Self {
            rows,
            cols,
            data: zeros.data,
        }
    }

    pub fn from(vec: Vec<Vec<f32>>) -> Self {
        // Assert that the vector is not empty.
        assert!(!vec.is_empty());

        let mut data = Vec::new();
        let rows = vec.len();
        let cols = vec[0].len();

        for row in 0..rows {
            let mut row_data = Vec::new();
            for col in 0..cols {
                row_data.push(Scalar::new(vec[row][col], false));
            }
            data.push(row_data);
        }

        Self { rows, cols, data }
    }

    // From a scalar creates a 1x1 tensor.
    pub fn scalar(scalar: f32) -> Self {
        Self {
            rows: 1,
            cols: 1,
            data: vec![vec![Scalar::new(scalar, false)]],
        }
    }

    // From a 1D array reference creates a 1xN 2dtensor
    pub fn row(vec: Vec<f32>) -> Self {
        Self::from(vec![vec.clone()])
    }

    pub fn col(vec: Vec<f32>) -> Self {
        Self::from(vec![vec.clone()]).transpose()
    }

    pub fn transpose(&self) -> Tensor2D {
        let mut ans = Self::zeros(self.cols, self.rows, false);

        for row in 0..self.rows {
            for col in 0..self.cols {
                ans.data[col][row] = self.data[row][col].clone();
            }
        }

        ans
    }

    pub fn backward(&self) -> Vec<Rc<RefCell<Data>>> {
        // Accumulate the nodes that need to be backpropagated
        let mut nodes: Vec<Rc<RefCell<Data>>> = Vec::new();

        for row in 0..self.rows {
            for col in 0..self.cols {
                // Extend the nodes with the backward of the current node
                nodes.extend(self.data[row][col].backward());
            }
        }

        nodes
    }

    pub fn nonlinear(tensor: &Self, activation: Activation) -> Tensor2D {
        let mut ans = Self::zeros(tensor.rows, tensor.cols, false);

        for row in 0..tensor.rows {
            for col in 0..tensor.cols {
                match activation {
                    Activation::Tanh => ans.data[row][col] = tensor.data[row][col].tanh(),
                    Activation::Sigmoid => ans.data[row][col] = tensor.data[row][col].sigmoid(),
                    Activation::ReLU => ans.data[row][col] = tensor.data[row][col].relu(),
                    Activation::Exp => ans.data[row][col] = tensor.data[row][col].exp(),
                }
            }
        }

        ans
    }

    pub fn tanh(&self) -> Tensor2D {
        Self::nonlinear(self, Activation::Tanh)
    }

    pub fn sigmoid(&self) -> Tensor2D {
        Self::nonlinear(self, Activation::Sigmoid)
    }

    pub fn relu(&self) -> Tensor2D {
        Self::nonlinear(self, Activation::ReLU)
    }

    pub fn exp(&self) -> Tensor2D {
        Self::nonlinear(self, Activation::Exp)
    }

    // The pow of a tensor is a tensor
    // To compute it we have to call pow on each element of the tensor
    pub fn pow(self, power: usize) -> Tensor2D {
        let mut ans = Self::zeros(self.rows, self.cols, false);

        for row in 0..self.rows {
            for col in 0..self.cols {
                ans.data[row][col] = self.data[row][col].pow(power);
            }
        }

        ans
    }
}
