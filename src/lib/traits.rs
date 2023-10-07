use crate::lib::grad::Scalar;
use crate::lib::grad::Node;
use crate::lib::tensor::Tensor2D;
use std::fmt::Display;
use std::ops::Deref;


impl Display for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.node.grad.get() {
            Some(grad) => {
                write!(f, "{:.3} [{:.3}]", self.value(), grad)
            }
            None => {
                write!(f, "{:.3} [None]", self.value())
            }
        }
    }
}

impl Deref for Scalar {
    type Target = Node;

    fn deref(&self) -> &Node {
        &self.node
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
