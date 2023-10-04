use crate::lib::grad::Scalar;
use crate::lib::tensor::Tensor2D;
use std::fmt::Display;
use std::rc::Rc;

impl Display for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value = {:.4}, Grad = {:.4}", self.value(), self.grad())
    }
}

impl Clone for Scalar {
    fn clone(&self) -> Self {
        Scalar {
            data: self.data.clone(),
        }
    }
}

impl Eq for Scalar {}

impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.data, &other.data)
    }
}

impl std::hash::Hash for Scalar {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.data).hash(state);
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

impl Clone for Tensor2D {
    fn clone(&self) -> Self {
        let mut data = Vec::new();

        for row in 0..self.rows {
            let mut row_data = Vec::new();
            for col in 0..self.cols {
                row_data.push(self.data[row][col].clone());
            }
            data.push(row_data);
        }

        Self {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}
