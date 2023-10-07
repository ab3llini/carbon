use std::cell::Cell;
use std::collections::HashSet;
use std::f64::consts::E;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub enum Operation {
    Add,
    Mul,
    Sub,
    Div,
}

#[derive(Debug, Clone)]
pub enum Activation {
    Exp,
    Tanh,
    Sigmoid,
    ReLU,
}

#[derive(Debug, Clone)]
pub enum Dependency {
    Single {
        scalar: Scalar,
        activation: Activation,
    },
    Double {
        lhs: Scalar,
        rhs: Scalar,
        op: Operation,
    },
}

#[derive(Debug, Clone)]
pub struct Node {
    pub val: Cell<f64>,
    pub grad: Cell<Option<f64>>,
    pub dep: Option<Dependency>,
}

#[derive(Debug, Clone)]
pub struct Scalar {
    pub node: Rc<Node>,
}

impl Scalar {
    pub fn new(value: f64) -> Self {
        Self {
            node: Rc::new(Node {
                val: Cell::new(value),
                grad: Cell::new(None),
                dep: None,
            }),
        }
    }

    pub fn grad(self: &Scalar) -> Option<f64> {
        self.grad.get()
    }

    pub fn value(self: &Scalar) -> f64 {
        self.val.get()
    }

    fn accumulate(self: &Scalar, value: f64) -> () {
        match self.grad.get() {
            Some(grad) => self.grad.set(Some(grad + value)),
            None => self.grad.set(Some(value)),
        }
    }

    pub fn compute(lhs: &Scalar, rhs: &Scalar, op: Operation) -> Self {
        let result: f64 = match op {
            Operation::Add => lhs.value() + rhs.value(),
            Operation::Mul => lhs.value() * rhs.value(),
            Operation::Sub => lhs.value() - rhs.value(),
            Operation::Div => lhs.value() / rhs.value(),
        };

        Self {
            node: Rc::new(Node {
                val: Cell::new(result),
                grad: Cell::new(None),
                dep: Some(Dependency::Double {
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                    op,
                }),
            }),
        }
    }

    pub fn tanh(self: &Scalar) -> Self {
        Self {
            node: Rc::new(Node {
                val: Cell::new(self.value().tanh()),
                grad: Cell::new(None),
                dep: Some(Dependency::Single {
                    scalar: self.clone(),
                    activation: Activation::Tanh,
                }),
            }),
        }
    }

    pub fn exp(self: &Scalar) -> Self {
        Self {
            node: Rc::new(Node {
                val: Cell::new(E.powf(self.value())),
                grad: Cell::new(None),
                dep: Some(Dependency::Single {
                    scalar: self.clone(),
                    activation: Activation::Exp,
                }),
            }),
        }
    }
    pub fn sigmoid(self: &Scalar) -> Self {
        Self {
            node: Rc::new(Node {
                val: Cell::new(1.0 / (1.0 + (-self.value()).exp())),
                grad: Cell::new(None),
                dep: Some(Dependency::Single {
                    scalar: self.clone(),
                    activation: Activation::Sigmoid,
                }),
            }),
        }
    }
    pub fn relu(self: &Scalar) -> Self {
        let val = {
            if self.value() > 0.0 {
                self.value()
            } else {
                0.0
            }
        };

        Self {
            node: Rc::new(Node {
                val: Cell::new(val),
                grad: Cell::new(None),
                dep: Some(Dependency::Single {
                    scalar: self.clone(),
                    activation: Activation::ReLU,
                }),
            }),
        }
    }

    pub fn pow(self: &Scalar, power: f64) -> Self {
        let mut ans = self.clone();
        for _ in 1..power as usize {
            ans = &ans * &self.clone();
        }
        ans
    }

    fn _backward(self: &Scalar) -> () {
        if let (Some(dep), Some(grad)) = (&self.dep, self.grad.get()) {
            // When we have an operation, we need to apply the chain rule to calculate the derivative.
            // The chain rule states that the derivative of a function f(g(x)) is f'(g(x)) * g'(x).

            match dep {
                Dependency::Double {
                    lhs,
                    rhs,
                    op: Operation::Add,
                } => {
                    // Addition means: f(x) = x + y, f'(x) = 1, f'(y) = 1
                    // So we just accumulate the gradient by 1.0 times the gradient of the output.
                    lhs.accumulate(grad * 1.0);
                    rhs.accumulate(grad * 1.0);
                }
                Dependency::Double {
                    lhs,
                    rhs,
                    op: Operation::Mul,
                } => {
                    // Multiplication means: f(x) = x * y, f'(x) = y, f'(y) = x
                    // So we just accumulate the gradient by the other parent times the gradient of the output.
                    lhs.accumulate(grad * rhs.value());
                    rhs.accumulate(grad * lhs.value());
                }
                Dependency::Double {
                    lhs,
                    rhs,
                    op: Operation::Sub,
                } => {
                    // Subtraction means: f(x) = x - y, f'(x) = 1, f'(y) = -1
                    // So we just accumulate the gradient by 1.0 or -1.0 times the gradient of the output.
                    lhs.accumulate(grad * 1.0);
                    rhs.accumulate(grad * -1.0);
                }
                Dependency::Double {
                    lhs,
                    rhs,
                    op: Operation::Div,
                } => {
                    // Division means: f(x) = x / y, f'(x) = 1 / y, f'(y) = -x / y^2
                    // So, for the left hand side, we accumulate by 1.0 divided by the right hand side times the gradient of the output.
                    // For the right hand side, we accumulate by -1.0 times the left hand side divided by the right hand side squared times the gradient of the output.
                    lhs.accumulate(grad * 1.0 / rhs.value());
                    rhs.accumulate(grad * -1.0 * lhs.value() / rhs.value().powi(2));
                }
                Dependency::Single {
                    scalar,
                    activation: Activation::Tanh,
                } => {
                    // Tanh means: f(x) = tanh(x), f'(x) = 1 - tanh(x)^2
                    // So, we set the gradient of the parent to 1.0 minus the parent squared times the gradient of the output.
                    scalar.accumulate(grad * (1.0 - self.value().powi(2)));
                }
                Dependency::Single {
                    scalar,
                    activation: Activation::Exp,
                } => {
                    // Exp means: f(x) = e^x, f'(x) = e^x
                    // So, we set the gradient of the parent to e^x times the gradient of the output.
                    scalar.accumulate(grad * self.value());
                }
                Dependency::Single {
                    scalar,
                    activation: Activation::Sigmoid,
                } => {
                    // Sigmoid means: f(x) = 1 / (1 + e^-x), f'(x) = f(x) * (1 - f(x))
                    // So, we set the gradient of the parent to f(x) times 1 - f(x) times the gradient of the output.
                    scalar.accumulate(grad * self.value() * (1.0 - self.value()));
                }
                Dependency::Single {
                    scalar,
                    activation: Activation::ReLU,
                } => {
                    // ReLU means: f(x) = max(0, x), f'(x) = 1 if x > 0, 0 otherwise
                    // So, we set the gradient of the parent to 1.0 if the parent is greater than 0.0, 0.0 otherwise.
                    if scalar.value() > 0.0 {
                        scalar.accumulate(grad * 1.0);
                    } else {
                        scalar.accumulate(grad * 0.0);
                    }
                }
            }
        }
    }

    fn topological(value: &Scalar, visited: &mut HashSet<usize>, stack: &mut Vec<Self>) {
        if !visited.contains(&value.ptr()) {
            // Insert the node into the visited set
            visited.insert(value.ptr());

            match &value.node.dep {
                Some(Dependency::Single {
                    scalar,
                    activation: _,
                }) => {
                    Self::topological(scalar, visited, stack);
                }
                Some(Dependency::Double { lhs, rhs, op: _ }) => {
                    Self::topological(lhs, visited, stack);
                    Self::topological(rhs, visited, stack);
                }
                None => {}
            }

            // Push the node onto the stack.
            // Make sure to do this operation after the recursive calls.
            stack.push(value.clone());
        }
    }

    // Returns the hash of the Rc pointer
    fn ptr(self: &Scalar) -> usize {
        Rc::as_ptr(&self.node) as usize
    }

    pub fn backward(self: &Scalar) {
        // Base strutures to sort the nodes topologically
        let mut visited: HashSet<usize> = HashSet::new();
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
