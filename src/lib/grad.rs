use std::cell::RefCell;
use std::collections::HashSet;
use std::f64::consts::E;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub enum Operation {
    Add,
    Mul,
    Sub,
    Div
}

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Exp,
    Tanh,
    Sigmoid,
    ReLU,
}

#[derive(Debug, Clone)]
pub enum Ancestor {
    Single { x: Scalar, activation: Activation },
    Double { lhs: Scalar, rhs: Scalar, operation: Operation },
}

#[derive(Debug, Clone)]
pub struct Node {
    pub value: f64,
    pub prev: Option<Ancestor>,
    pub grad: f64,
}

#[derive(Debug)]
pub struct Scalar {
    pub data: Rc<RefCell<Node>>,
}

impl Scalar {
    pub fn new(value: f64) -> Self {
        Self {
            data: Rc::new(RefCell::new(Node {
                value,
                prev: None,
                grad: 0.0,
            })),
        }
    }

    pub fn grad(&self) -> f64 {
        self.data.borrow().grad
    }

    pub fn value(&self) -> f64 {
        self.data.borrow().value
    }

    pub fn set_value(&self, value: f64) -> () {
        self.data.borrow_mut().value = value;
    }

    fn accumulate(&self, value: f64) -> () {
        self.data.borrow_mut().grad += value
    }

    pub fn compute(lhs: Self, rhs: Self, op: Operation) -> Self {
        let value: f64 = match op {
            Operation::Add => lhs.value() + rhs.value(),
            Operation::Mul => lhs.value() * rhs.value(),
            Operation::Sub => lhs.value() - rhs.value(),
            Operation::Div => lhs.value() / rhs.value(),
        };
        let node: Node = Node {
            value,
            prev: Some(Ancestor::Double { lhs, rhs, operation: op }),
            grad: 0.0,
        };
        Scalar {
            data: Rc::new(RefCell::new(node)),
        }
    }

    fn nonlinear(x: Self, activation: Activation) -> Self {
        let value: f64 = match activation {
            Activation::Exp => E.powf(x.value()),
            Activation::Tanh => x.value().tanh(),
            Activation::Sigmoid => 1.0 / (1.0 + (-x.value()).exp()),
            Activation::ReLU => {
                if x.value() > 0.0 {
                    x.value()
                } else {
                    0.0
                }
            }
        };
        let node: Node = Node {
            value,
            prev: Some(Ancestor::Single { x, activation}),
            grad: 0.0,
        };
        Scalar {
            data: Rc::new(RefCell::new(node)),
        }
    }

    pub fn tanh(self) -> Self {
        Self::nonlinear(self, Activation::Tanh)
    }

    pub fn exp(self) -> Self {
        Self::nonlinear(self, Activation::Exp)
    }
    pub fn sigmoid(self) -> Self {
        Self::nonlinear(self, Activation::Sigmoid)
    }
    pub fn relu(self) -> Self {
        Self::nonlinear(self, Activation::ReLU)
    }

    // The pow of a scalar is a scalar
    // x^2 is like x*x and x^3 is like x*x*x
    // This function takes a scalar and computes self^power multiplying self by itself power times
    pub fn pow(self, power: f64) -> Self {
        let mut ans = self.clone();
        for _ in 1..power as usize {
            ans = &ans * &self.clone();
        }
        ans
    }

    

    fn _backward(self) -> () {
        if let Some(prev) = &self.data.borrow().prev {

            // When we have an operation, we need to apply the chain rule to calculate the derivative.
            // The chain rule states that the derivative of a function f(g(x)) is f'(g(x)) * g'(x).

            match prev {

                Ancestor::Double { lhs, rhs, operation:Operation::Add } => {
                    // Addition means: f(x) = x + y, f'(x) = 1, f'(y) = 1
                    // So we just accumulate the gradient by 1.0 times the gradient of the output.
                    lhs.accumulate(self.grad() * 1.0);
                    rhs.accumulate(self.grad() * 1.0);
                }
                Ancestor::Double { lhs, rhs, operation:Operation::Mul } => {
                    // Multiplication means: f(x) = x * y, f'(x) = y, f'(y) = x
                    // So we just accumulate the gradient by the other parent times the gradient of the output.
                    lhs.accumulate(self.grad() * rhs.value());
                    rhs.accumulate(self.grad() * lhs.value());
                }
                Ancestor::Double { lhs, rhs, operation:Operation::Sub } => {
                    // Subtraction means: f(x) = x - y, f'(x) = 1, f'(y) = -1
                    // So we just accumulate the gradient by 1.0 or -1.0 times the gradient of the output.
                    lhs.accumulate(self.grad() * 1.0);
                    rhs.accumulate(self.grad() * -1.0);
                }
                Ancestor::Double { lhs, rhs, operation:Operation::Div } => {
                    // Division means: f(x) = x / y, f'(x) = 1 / y, f'(y) = -x / y^2
                    // So, for the left hand side, we accumulate by 1.0 divided by the right hand side times the gradient of the output.
                    // For the right hand side, we accumulate by -1.0 times the left hand side divided by the right hand side squared times the gradient of the output.
                    lhs.accumulate(self.grad() * 1.0 / rhs.value());
                    rhs.accumulate(self.grad() * -1.0 * lhs.value() / rhs.value().powi(2));
                }
                Ancestor::Single { x, activation:Activation::Tanh } => {
                    // Tanh means: f(x) = tanh(x), f'(x) = 1 - tanh(x)^2
                    // So, we set the gradient of the parent to 1.0 minus the parent squared times the gradient of the output.
                    x.accumulate(self.grad() * (1.0 - self.value().powi(2)));
                }
                Ancestor::Single { x, activation:Activation::Exp } => {
                    // Exp means: f(x) = e^x, f'(x) = e^x
                    // So, we set the gradient of the parent to e^x times the gradient of the output.
                    x.accumulate(self.grad() * self.value());
                }
                Ancestor::Single { x, activation:Activation::Sigmoid } => {
                    // Sigmoid means: f(x) = 1 / (1 + e^-x), f'(x) = f(x) * (1 - f(x))
                    // So, we set the gradient of the parent to f(x) times 1 - f(x) times the gradient of the output.
                    x.accumulate(self.grad() * self.value() * (1.0 - self.value()));
                }
                Ancestor::Single { x, activation:Activation::ReLU } => {
                    // ReLU means: f(x) = max(0, x), f'(x) = 1 if x > 0, 0 otherwise
                    // So, we set the gradient of the parent to 1.0 if the parent is greater than 0.0, 0.0 otherwise.
                    if x.value() > 0.0 {
                        x.accumulate(self.grad() * 1.0);
                    } else {
                        x.accumulate(self.grad() * 0.0);
                    }
                }
            }
        }
    }

    fn topological(value: &Self, visited: &mut HashSet<Scalar>, stack: &mut Vec<Scalar>) {
        if !visited.contains(value) {
            // Insert the node into the visited set
            visited.insert(value.clone());

            if let Some(prev) = &value.data.borrow().prev {
                match prev {
                    Ancestor::Single { x, activation: _ } => {
                        Self::topological(&x, visited, stack);
                    }
                    Ancestor::Double { lhs, rhs, operation: _ } => {
                        Self::topological(&lhs, visited, stack);
                        Self::topological(&rhs, visited, stack);
                    }
                }
            }

            // Push the node onto the stack.
            // Make sure to do this operation after the recursive calls.
            stack.push(value.clone());
        }
    }

    pub fn backward(&self) {
        // Base strutures to sort the nodes topologically
        let mut visited: HashSet<Scalar> = HashSet::new();
        let mut stack: Vec<Scalar> = Vec::new();

        // Sort the nodes topologically
        Self::topological(self, &mut visited, &mut stack);

        // Reverse the stack, since we want to backpropagate from the output to the input
        stack.reverse();

        // Set the gradient of the output to 1.0
        self.accumulate(1.0);

        // Backpropagate the gradient
        for node in stack {
            node._backward();
        }
    }

}
