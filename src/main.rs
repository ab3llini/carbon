mod lib {
    pub mod grad;
    pub mod loss;
    pub mod macros;
    pub mod nn;
    pub mod ops;
    pub mod tensor;
    pub mod traits;
}

use std::vec::Vec;

use lib::grad::Activation;
use lib::grad::Scalar;
use lib::loss;
use lib::nn::MLP;
use lib::tensor::Tensor2D;

fn main() {
    let nn: MLP = MLP::new(vec![3, 4, 4, 1], Activation::Tanh);

    let x_train = vec![
        Tensor2D::row(vec![2.0, 3.0, -1.0]),
        Tensor2D::row(vec![3.0, -1.0, 0.5]),
        Tensor2D::row(vec![0.5, 1.0, 1.0]),
        Tensor2D::row(vec![1.0, 1.0, -1.0])
    ];

    let y_train: Vec<Tensor2D> = vec![
        Tensor2D::scalar(1.0),
        Tensor2D::scalar(-1.0),
        Tensor2D::scalar(-1.0),
        Tensor2D::scalar(1.0)
    ];

    // Gradient Descent
    let lr: f32 = 0.05;
    let epochs: usize = 1000;
    let log_every: usize = 10;

    for i in 0..epochs {
        // Temporary vector to store predictions
        let mut preds: Vec<Tensor2D> = vec![];

        // For each element in our small dataset, perform a forward pass
        for input in x_train.iter() {
            preds.push(nn.forward(input));
        }

        // Compute the loss with MSE
        let loss: Scalar = loss::mse(&preds, &y_train);

        if loss.val().abs() < 0.001 {
            println!("Converged in {} epochs", i);
            break;
        }

        if log_every != 0 && i % log_every == 0 {
            println!("Loss: {:4}", loss.val());
        }

        // Zero the gradients
        for param in nn.params() {
            let mut data = param.borrow_mut();
            data.grad = 0.0;
        }

        // Backpropagate gradients
        loss.backward();

        // Update parameters
        for param in nn.params() {
            let mut data = param.borrow_mut();
            data.val += -lr * data.grad;
        }
    }

    // Print preds
    print!("Preds: ");
    for input in x_train.iter() {
        for pred in nn.forward(input).data[0].iter() {
            print!("| {}", pred.val());
        }
    }
    println!(" |");
}
