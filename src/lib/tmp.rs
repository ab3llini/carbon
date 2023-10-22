mod lib {
    pub mod grad;
    pub mod loss;
    pub mod macros;
    pub mod nn;
    pub mod ops;
    pub mod tensor;
    pub mod traits;
}

use lib::grad::Activation;
use lib::grad::Scalar;
use lib::loss;
use lib::nn::MLP;
use lib::tensor::Tensor2D;
use std::vec;

fn main() {
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
    let lr: f32 = 0.0001;
    let epochs: usize = 1000;

    for i in 0..epochs {
        // Temporary vector to store predictions
        let mut preds: vec::Vec<Tensor2D> = vec![];

        // For each element in our small dataset, perform a forward pass
        for input in x_train.iter() {
            preds.push(nn.forward(input));
        }

        // Compute the loss with MSE
        let loss: Scalar = loss::mse(&preds, &y_train);

        if loss.value().abs() < 0.001 {
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

    // Print preds
    print!("Preds: ");
    for input in x_train.iter() {
        for pred in nn.forward(input).data[0].iter() {
            print!("| {}", pred.value());
        }
    }
    print!(" |");
}
