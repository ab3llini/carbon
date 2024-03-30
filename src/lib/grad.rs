use crate::lib::ops::Operation;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

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
        prev: Rc<RefCell<Data>>,
        activation: Activation,
    },
    Double {
        lhs: Rc<RefCell<Data>>,
        rhs: Rc<RefCell<Data>>,
        op: Operation,
    },
}
#[derive(Debug, Clone)]
pub struct Data {
    pub val: f32,
    pub grad: f32,
    pub dep: Option<Dependency>,
    pub requires_grad: bool,
}

pub trait Nonlinear {
    fn exp(&self) -> Self;
    fn tanh(&self) -> Self;
    fn sigmoid(&self) -> Self;
    fn relu(&self) -> Self;
    fn pow(&self, power: usize) -> Self;
}

#[derive(Debug, Clone)]
pub struct Scalar {
    pub data: Rc<RefCell<Data>>,
}

impl Data {
    pub fn hash(rc: Rc<RefCell<Data>>) -> usize {
        Rc::as_ptr(&rc) as usize
    }

    pub fn backward(rc: Rc<RefCell<Data>>) -> () {
        let (dep, grad, val) = (&rc.borrow().dep, rc.borrow().grad, rc.borrow().val);

        match dep {
            Some(Dependency::Double { lhs, rhs, op }) => {
                // When we have an operation, we need to apply the chain rule to calculate the derivative.
                // The chain rule states that the derivative of a function f(g(x)) is f'(g(x)) * g\"(x).

                match op {
                    Operation::Add => {
                        // Addition means: f(x) = x + y, f'(x) = 1, f'(y) = 1
                        // So we just accumulate the gradient by 1.0 times the gradient of the output.
                        if lhs.as_ptr() == rhs.as_ptr() {
                            lhs.borrow_mut().grad += grad * 2.0;
                        } else {
                            lhs.borrow_mut().grad += grad * 1.0;
                            rhs.borrow_mut().grad += grad * 1.0;
                        }
                    }
                    Operation::Sub => {
                        // Subtraction means: f(x) = x - y, f'(x) = 1, f'(y) = -1
                        // So we just accumulate the gradient by 1.0 or -1.0 times the gradient of the output.
                        if lhs.as_ptr() == rhs.as_ptr() {
                            // Do nothing since we would sum and subtract the same quantity
                        } else {
                            lhs.borrow_mut().grad += grad * 1.0;
                            rhs.borrow_mut().grad += grad * -1.0;
                        }
                    }
                    Operation::Mul => {
                        // Multiplication means: f(x) = x * y, f'(x) = y, f'(y) = x
                        // So we just accumulate the gradient by the other parent times the gradient of the output.
                        if lhs.as_ptr() == rhs.as_ptr() {
                            let mut data_ref = lhs.borrow_mut();
                            data_ref.grad += 2.0 * grad * data_ref.val;
                        } else {
                            let (mut lhs_ref, mut rhs_ref) = (lhs.borrow_mut(), rhs.borrow_mut());

                            lhs_ref.grad += grad * rhs_ref.val;
                            rhs_ref.grad += grad * lhs_ref.val;
                        }
                    }
                    Operation::Div => {
                        // Division means: f(x) = x / y, f'(x) = 1 / y, f'(y) = -x / y^2
                        // So, for the left hand side, we accumulate by 1.0 divided by the right hand side times the gradient of the output.
                        // For the right hand side, we accumulate by -1.0 times the left hand side divided by the right hand side squared times the gradient of the output.
                        if lhs.as_ptr() == rhs.as_ptr() {
                            let mut data_ref = lhs.borrow_mut();
                            data_ref.grad += grad / data_ref.val;
                            data_ref.grad += (grad * -1.0 * data_ref.val) / data_ref.val.powi(2);
                        } else {
                            let (mut lhs_ref, mut rhs_ref) = (lhs.borrow_mut(), rhs.borrow_mut());

                            lhs_ref.grad += grad / rhs_ref.val;
                            rhs_ref.grad += (grad * -1.0 * lhs_ref.val) / rhs_ref.val.powi(2);
                        }
                    }
                }
            }

            Some(Dependency::Single { prev, activation }) => {
                match activation {
                    Activation::Tanh => {
                        // Tanh means: f(x) = tanh(x), f'(x) = 1 - tanh(x)^2
                        // So, we set the gradient of the parent to 1 - tanh(x)^2 times the gradient of the output.
                        prev.borrow_mut().grad += grad * (1.0 - val.powi(2));
                    }
                    Activation::Exp => {
                        // Exp means: f(x) = e^x, f'(x) = e^x
                        // So, we set the gradient of the parent to e^x times the gradient of the output.
                        prev.borrow_mut().grad += grad * val;
                    }
                    Activation::Sigmoid => {
                        // Sigmoid means: f(x) = 1 / (1 + e^-x), f'(x) = f(x) * (1 - f(x))
                        // So, we set the gradient of the parent to f(x) * (1 - f(x)) times the gradient of the output.
                        prev.borrow_mut().grad += grad * val * (1.0 - val);
                    }
                    Activation::ReLU => {
                        // ReLU means: f(x) = max(0, x), f'(x) = 1 if x > 0, 0 otherwise
                        // So, we set the gradient of the parent to 1 if x > 0, 0 otherwise times the gradient of the output.

                        prev.borrow_mut().grad += grad * (if val > 0.0 { 1.0 } else { 0.0 });
                    }
                }
            }

            None => (),
        }
    }
}

fn rc_2_str(rc: Rc<RefCell<Data>>) -> String {
    let hex_string = format!("{:p}", Rc::as_ptr(&rc));
    let clean_hex_string = hex_string.trim_start_matches("0x").to_uppercase();
    // ADD NODE_ at the beginning
    format!("NODE_{}", clean_hex_string)
}

impl Scalar {
    pub fn new(value: f32, requires_grad: bool) -> Self {
        Self {
            data: Rc::new(
                RefCell::new(Data {
                    val: value,
                    grad: 0.0,
                    dep: None,
                    requires_grad: requires_grad,
                })
            ),
        }
    }

    pub fn val(self: &Scalar) -> f32 {
        self.data.borrow().val
    }

    pub fn grad(self: &Scalar) -> f32 {
        self.data.borrow().grad
    }

    fn topological(
        data: Rc<RefCell<Data>>,
        visited: &mut HashSet<usize>,
        stack: &mut Vec<Rc<RefCell<Data>>>
    ) {
        let hash: usize = Data::hash(Rc::clone(&data));

        if data.borrow().requires_grad && !visited.contains(&hash) {
            // Insert the node into the visited set
            visited.insert(hash);

            match &data.borrow().dep {
                Some(Dependency::Single { prev, activation }) => {
                    // println!("{} -> {} [label=\"{}\"];", rc_2_str(Rc::clone(&data)), rc_2_str(Rc::clone(prev)), activation);
                    Self::topological(Rc::clone(prev), visited, stack);
                }
                Some(Dependency::Double { lhs, rhs, op }) => {
                    // println!("{} -> {} [label=\"{}\"];", rc_2_str(Rc::clone(&data)), rc_2_str(Rc::clone(lhs)), op);
                    Self::topological(Rc::clone(lhs), visited, stack);
                    // println!("{} -> {} [label=\"{}\"];", rc_2_str(Rc::clone(&data)), rc_2_str(Rc::clone(rhs)), op);
                    Self::topological(Rc::clone(rhs), visited, stack);
                }
                None => (),
            }

            // Push the node onto the stack.
            // Make sure to do this operation after the recursive calls.
            stack.push(Rc::clone(&data));
        }
    }

    pub fn backward(self: &Scalar) -> Vec<Rc<RefCell<Data>>> {
        // Base strutures to sort the nodes topologically
        let mut visited: HashSet<usize> = HashSet::new();
        let mut stack: Vec<Rc<RefCell<Data>>> = Vec::new();

        // Sort the nodes topologically
        Self::topological(Rc::clone(&self.data), &mut visited, &mut stack);

        // Reverse the stack, since we want to backpropagate from the output to the input
        stack.reverse();

        // Set the gradient of the output to 1.0
        self.data.borrow_mut().grad = 1.0;

        // Backpropagate the gradient
        for node in &stack {
            // println!("{} [label=\"V:{:.2}\\nG: {:.2}\\nRG={}\"];", rc_2_str(Rc::clone(node)), node.borrow().val, node.borrow().grad, node.borrow().requires_grad);
            Data::backward(Rc::clone(node));
        }

        stack
    }
}

impl Nonlinear for Scalar {
    fn tanh(&self) -> Self {
        Self {
            data: Rc::new(
                RefCell::new(Data {
                    val: self.data.borrow().val.tanh(),
                    grad: 0.0,
                    dep: Some(Dependency::Single {
                        prev: Rc::clone(&self.data),
                        activation: Activation::Tanh,
                    }),
                    requires_grad: self.data.borrow().requires_grad,
                })
            ),
        }
    }
    fn exp(&self) -> Self {
        Self {
            data: Rc::new(
                RefCell::new(Data {
                    val: self.data.borrow().val.exp(),
                    grad: 0.0,
                    dep: Some(Dependency::Single {
                        prev: Rc::clone(&self.data),
                        activation: Activation::Exp,
                    }),
                    requires_grad: self.data.borrow().requires_grad,
                })
            ),
        }
    }
    fn sigmoid(&self) -> Self {
        Self {
            data: Rc::new(
                RefCell::new(Data {
                    val: 1.0 / (1.0 + (-self.data.borrow().val).exp()),
                    grad: 0.0,
                    dep: Some(Dependency::Single {
                        prev: Rc::clone(&self.data),
                        activation: Activation::Sigmoid,
                    }),
                    requires_grad: self.data.borrow().requires_grad,
                })
            ),
        }
    }
    fn relu(&self) -> Self {
        let val = {
            if self.data.borrow().val > 0.0 { self.data.borrow().val } else { 0.0 }
        };
        Self {
            data: Rc::new(
                RefCell::new(Data {
                    val,
                    grad: 0.0,
                    dep: Some(Dependency::Single {
                        prev: Rc::clone(&self.data),
                        activation: Activation::ReLU,
                    }),
                    requires_grad: self.data.borrow().requires_grad,
                })
            ),
        }
    }
    fn pow(&self, power: usize) -> Self {
        let mut ans = self.clone();
        for _ in 1..power as usize {
            ans = &ans * &self.clone();
        }
        ans
    }
}
