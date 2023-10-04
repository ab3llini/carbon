use crate::lib::grad::Activation;
use crate::lib::tensor::Tensor2D;

use super::grad::Scalar;

// Neuron
pub struct Neuron {
    pub weights: Tensor2D,
    pub bias: Tensor2D,
    activation: Activation,
}

impl Neuron {
    pub fn new(size: usize, activation: Activation) -> Self {
        Self {
            weights: Tensor2D::xavier(1, size),
            bias: Tensor2D::xavier(1, 1),
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

    pub fn params(&self) -> Vec<Scalar> {
        let mut params: Vec<Scalar> = Vec::new();
        for row in self.weights.data.iter() {
            for scalar in row.iter() {
                params.push(scalar.clone());
            }
        }
       for row in self.bias.data.iter() {
            for scalar in row.iter() {
                params.push(scalar.clone());
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
            neurons.push(Neuron::new(in_size, activation));
        }
        Self { neurons }
    }

    pub fn forward(&self, input: &Tensor2D) -> Tensor2D {
        let mut output = Tensor2D::zeros(1, self.neurons.len());
        for (i, neuron) in self.neurons.iter().enumerate() {
            output.data[0][i] = neuron.forward(input).data[0][0].clone();
        }
        output
    }

    pub fn params(&self) -> Vec<Scalar> {
        let mut params = Vec::new();
        for neuron in self.neurons.iter() {
            params.append(&mut neuron.params());
        }
        params
    }
}

// MLP
pub struct MLP {
    pub layers: Vec<Layer>,
}

impl MLP {
    pub fn new(sizes: Vec<usize>, activation: Activation) -> Self {
        let mut layers = Vec::new();
        for i in 0..sizes.len() - 1 {
            layers.push(Layer::new(sizes[i], sizes[i + 1], activation));
        }
        Self { layers }
    }

    pub fn forward(&self, input: &Tensor2D) -> Tensor2D {
        let mut output = input.clone();
        for layer in self.layers.iter() {
            output = layer.forward(&output);
        }
        output
    }

    pub fn params(&self) -> Vec<Scalar> {
        let mut params = Vec::new();
        for layer in self.layers.iter() {
            params.append(&mut layer.params());
        }
        params
    }
}
