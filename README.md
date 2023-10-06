# Carbon: A Rust-Based Micrograd Implementation

[![Rust](https://img.shields.io/badge/Language-Rust-orange.svg)](https://www.rust-lang.org/) [![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-blue.svg)](https://github.com/yourusername/carbon)

Welcome to Carbon! 🚀 
Carbon is a fully functional differentiation engine for 2D tensors and scalar operations, written in Rust. It's inspired by [micrograd](https://github.com/karpathy/micrograd).
This project is a labor of love, born out of the desire to learn Rust while delving into the exciting world of deep learning. While it's not a full-fledged deep learning framework, it actually works to train dummy neural nets! 🎉

## Project Highlights
- 🤖 **Neural Net Layer:** Carbon features a neural net layer, inspired by micrograd, that includes neurons, layers, and multi-layer perceptrons (MLP). It's perfect for experimenting with basic neural networks.
- 🧮 **Fully Functional Differentiation Engine**: Carbon boasts a fully functional differentiation engine, allowing you to compute gradients for your custom neural network architectures.
- 📦 **No External Dependencies**: Carbon takes pride in being self-contained. It doesn't rely on external libraries, making it a lightweight and pure Rust experience.
- 🧬 **Restricted Capabilities:** Carbon is intentionally limited, supporting only 2D tensors and scalar operations. Don't expect it to compete with heavy-duty deep learning libraries; instead, think of it as a learning tool to explore Rust.

## Getting Started

To get started with Carbon, you can clone the repository and start exploring. Remember, this project is all about learning and having fun with Rust!

```bash
git clone https://github.com/yourusername/carbon.git
cd carbon
cargo run
```

Feel free to experiment, tinker, and expand on what Carbon offers. Rust is a fantastic language for low-level systems programming and building robust applications, so let your creativity run wild!

Happy coding! 🚀🦀

## How it works

Here's a glimpse of what Carbon offers:

#### Scalars
```rust
let a = Scalar::new(1.0);
let b = Scalar::new(2.0);
let c = &a + &b;
let d = c.relu().exp();

d.backward();

// Gradient of b with respect to d
print!("b: {}", b.grad()); // 20.085
```

#### Tensors

```rust
let a = Tensor2D::row(vec![1.0, 1.0, 1.0]); // 1x3
let b = Tensor2D::uniform(3, 3); // 3x3

let c = &a * &b; // 1x3
let d = &a + &c; // 1x3
let e = d.pow(2.0).tanh(); //1x3

// Compute gradients with backpropagation
e.backward();

// Values & Gradient of c with respect to e
 println!("c: {}", c);
// Value = 2.092, Grad = 0 | Value = -0.585, Grad = 0.805 | Value = -0.418, Grad = 1.039
```

#### MLP

```rust
let nn: MLP = MLP::new(vec![3, 4, 4, 1], Activation::Tanh);

let x_train = vec![
    Tensor2D::row(vec![2.0, 3.0, -1.0]),
    Tensor2D::row(vec![3.0, -1.0, 0.5]),
    Tensor2D::row(vec![0.5, 1.0, 1.0]),
    Tensor2D::row(vec![1.0, 1.0, -1.0]),
];

let y_train: Vec<Tensor2D> = vec![
    Tensor2D::scalar(1.0),
    Tensor2D::scalar(-1.0),
    Tensor2D::scalar(-1.0),
    Tensor2D::scalar(1.0),
];

// Gradient Descent
let lr: f64 = 0.000001;
let epochs: usize = 10000;

for i in 0..epochs {
    // Temporary vector to store predictions
    let mut preds: vec::Vec<Tensor2D> = vec![];

    // For each element in our small dataset, perform a forward pass
    for input in x_train.iter() {
        preds.push(nn.forward(input));
    }

    // Compute the loss with MSE
    let loss: Scalar = loss::mse(&preds, &y_train);

    if loss.value().abs() < 0.0001 {
        println!("Converged in {} epochs", i);
        break;
    }

    // Backpropagate gradients
    loss.backward();

    // Update parameters
    for param in nn.params() {
        param.set_value(param.value() - lr * param.grad())
    }

    println!("Loss: {:?}", loss.value());
}
```

## Contributing

We welcome contributions from the Rust community. If you have ideas, bug fixes, or improvements, please open an issue or submit a pull request. Let's build something awesome together! 🤝

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.