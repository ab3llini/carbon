use std::cell::RefCell;
use std::rc::Rc;

use crate::lib::grad::Activation;
use crate::lib::tensor::Tensor2D;

use super::grad::Data;
use super::grad::Scalar;

// use super::loss;

// Neuron
pub struct Neuron {
    pub weights: Tensor2D,
    pub bias: Tensor2D,
    activation: Activation,
}

impl Neuron {
    pub fn new(size: usize, activation: Activation) -> Self {
        Self {
            weights: Tensor2D::xavier(1, size, true),
            bias: Tensor2D::xavier(1, 1, true),
            activation,
        }
    }

    // Input shape is (1, n)
    // Weights shape is (n, 1)
    pub fn forward(&self, input: &Tensor2D) -> Tensor2D {
        let out = &(input * &self.weights.transpose()) + &self.bias;
        match self.activation {
            Activation::Exp => out.exp(),
            Activation::Tanh => out.tanh(),
            Activation::Sigmoid => out.sigmoid(),
            Activation::ReLU => out.relu(),
        }
    }

    pub fn params(&self) -> Vec<Rc<RefCell<Data>>> {
        let mut params: Vec<Rc<RefCell<Data>>> = Vec::new();

        for row in self.weights.data.iter() {
            for scalar in row.iter() {
                params.push(Rc::clone(&scalar.data));
            }
        }
        for row in self.bias.data.iter() {
            for scalar in row.iter() {
                params.push(Rc::clone(&scalar.data));
            }
        }
        params
    }
}

// Layer
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(in_size: usize, out_size: usize, activation: Activation) -> Self {
        let mut neurons = Vec::new();
        for _ in 0..out_size {
            neurons.push(Neuron::new(in_size, activation.clone()));
        }
        Self { neurons }
    }

    pub fn forward(&self, input: &Tensor2D) -> Tensor2D {
        let mut output = Tensor2D::zeros(1, self.neurons.len(), false);
        for (i, neuron) in self.neurons.iter().enumerate() {
            output.data[0][i] = neuron.forward(input).data[0][0].clone();
        }
        output
    }

    pub fn params(&self) -> Vec<Rc<RefCell<Data>>> {
        let mut params: Vec<Rc<RefCell<Data>>> = Vec::new();
        for neuron in self.neurons.iter() {
            params.append(&mut neuron.params());
        }
        params
    }
}

// MLP
pub struct MLP {
    pub layers: Vec<Layer>,
    pub topological: Option<Vec<Rc<RefCell<Data>>>>,
}

impl MLP {
    pub fn new(sizes: Vec<usize>, activation: Activation) -> Self {
        let mut layers = Vec::new();
        for i in 0..sizes.len() - 1 {
            layers.push(Layer::new(sizes[i], sizes[i + 1], activation.clone()));
        }
        Self {
            layers,
            topological: None,
        }
    }

    pub fn forward(&self, input: &Tensor2D) -> Tensor2D {
        let mut output: Tensor2D = input.clone();

        for layer in self.layers.iter() {
            output = layer.forward(&output)
        }

        output
    }

    pub fn params(&self) -> Vec<Rc<RefCell<Data>>> {
        let mut params: Vec<Rc<RefCell<Data>>> = Vec::new();
        for layer in self.layers.iter() {
            params.append(&mut layer.params());
        }
        params
    }

    pub fn backward(&mut self, loss: &Scalar) {
        match &self.topological {
            Some(order) => {
                for node in order {
                    Data::backward(Rc::clone(node));
                }
            }
            None => {
                self.topological = Some(loss.backward());
            }
        }
    }
}
