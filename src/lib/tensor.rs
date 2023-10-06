use crate::lib::grad::Activation;
use crate::lib::grad::Scalar;
use rand::distributions::Uniform;
use rand::Rng;
use std::vec::Vec;

#[derive(Debug)]
pub struct Tensor2D {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<Scalar>>,
}

impl Tensor2D {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        // Do not initialize data with something like this:
        // vec![vec![Scalar::new(0.0); cols]; rows],
        // This leads to a huge bug where all the scalars in the tensor point to the same memory address.

        let data = {
            let mut data = Vec::new();
            for _ in 0..rows {
                let mut row = Vec::new();
                for _ in 0..cols {
                    row.push(Scalar::new(0.0));
                }
                data.push(row);
            }
            data
        };

        Self { rows, cols, data }
    }

    pub fn uniform(rows: usize, cols: usize) -> Self {
        let zeros = Self::zeros(rows, cols);
        let mut rng = rand::thread_rng();
        let side = Uniform::new(-1.0, 1.0);

        for row in 0..zeros.rows {
            for col in 0..zeros.cols {
                zeros.data[row][col].set_value(rng.sample(side));
            }
        }

        Self {
            rows,
            cols,
            data: zeros.data,
        }
    }

    pub fn xavier(rows: usize, cols: usize) -> Self {
        let zeros = Self::zeros(rows, cols);
        let mut rng = rand::thread_rng();
        let side = Uniform::new(-1.0, 1.0);

        for row in 0..zeros.rows {
            for col in 0..zeros.cols {
                zeros.data[row][col].set_value(rng.sample(side) / (rows as f64).sqrt());
            }
        }

        Self {
            rows,
            cols,
            data: zeros.data,
        }
    }

    pub fn from(vec: Vec<Vec<f64>>) -> Self {
        // Assert that the vector is not empty.
        assert!(!vec.is_empty());

        let mut data = Vec::new();
        let rows = vec.len();
        let cols = vec[0].len();

        for row in 0..rows {
            let mut row_data = Vec::new();
            for col in 0..cols {
                row_data.push(Scalar::new(vec[row][col]));
            }
            data.push(row_data);
        }

        Self { rows, cols, data }
    }

    // From a scalar creates a 1x1 tensor.
    pub fn scalar(scalar: f64) -> Self {
        Self {
            rows: 1,
            cols: 1,
            data: vec![vec![Scalar::new(scalar)]],
        }
    }

    // From a 1D array reference creates a 1xN 2dtensor
    pub fn row(vec: Vec<f64>) -> Self {
        Self::from(vec![vec.clone()])
    }

    pub fn col(vec: Vec<f64>) -> Self {
        Self::from(vec![vec.clone()]).transpose()
    }

    pub fn transpose(&self) -> Tensor2D {
        let mut ans = Self::zeros(self.cols, self.rows);

        for row in 0..self.rows {
            for col in 0..self.cols {
                ans.data[col][row] = self.data[row][col].clone();
            }
        }

        ans
    }

    pub fn backward(&self) -> () {
        for row in 0..self.rows {
            for col in 0..self.cols {
                self.data[row][col].backward();
            }
        }
    }

    pub fn nonlinear(tensor: Self, activation: Activation) -> Tensor2D {
        let mut ans = Self::zeros(tensor.rows, tensor.cols);

        for row in 0..tensor.rows {
            for col in 0..tensor.cols {
                match activation {
                    Activation::Tanh => ans.data[row][col] = tensor.data[row][col].clone().tanh(),
                    Activation::Sigmoid => {
                        ans.data[row][col] = tensor.data[row][col].clone().sigmoid()
                    }
                    Activation::ReLU => ans.data[row][col] = tensor.data[row][col].clone().relu(),
                    Activation::Exp => ans.data[row][col] = tensor.data[row][col].clone().exp(),
                }
            }
        }

        ans
    }

    pub fn tanh(self) -> Tensor2D {
        Self::nonlinear(self, Activation::Tanh)
    }

    pub fn sigmoid(self) -> Tensor2D {
        Self::nonlinear(self, Activation::Sigmoid)
    }

    pub fn relu(self) -> Tensor2D {
        Self::nonlinear(self, Activation::ReLU)
    }

    pub fn exp(self) -> Tensor2D {
        Self::nonlinear(self, Activation::Exp)
    }

    // The pow of a tensor is a tensor
    // To compute it we have to call pow on each element of the tensor
    pub fn pow(self, power: f64) -> Tensor2D {
        let mut ans = Self::zeros(self.rows, self.cols);

        for row in 0..self.rows {
            for col in 0..self.cols {
                ans.data[row][col] = self.data[row][col].clone().pow(power);
            }
        }

        ans
    }
}
