use crate::lib::grad::Scalar;
use crate::lib::grad::Activation;
use crate::lib::ops::Operation;
use crate::lib::tensor::Tensor2D;
use std::fmt::Display;

impl Display for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.4} [{:.4}]", self.data.borrow().val, self.data.borrow().grad)
    }
}
impl Display for Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Activation::Exp => write!(f, "exp"),
            Activation::Tanh => write!(f, "tanh"),
            Activation::Sigmoid => write!(f, "sigmoid"),
            Activation::ReLU => write!(f, "relu"),
        }
    }
}

impl Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operation::Add => write!(f, "+"),
            Operation::Sub => write!(f, "-"),
            Operation::Mul => write!(f, "*"),
            Operation::Div => write!(f, "/"),
        }
    }
}

impl Display for Tensor2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ans = String::new();

        for row in 0..self.rows {
            ans.push_str("| ");
            for col in 0..self.cols {
                ans.push_str(&format!("{} | ", self.data[row][col]));
            }
            ans.push_str("\n");
        }

        write!(f, "{}", ans)
    }
}
