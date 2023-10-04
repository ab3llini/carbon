use crate::lib::grad::Scalar;
use crate::lib::tensor::Tensor2D;

// MSE loss
pub fn mse(y_pred: &Vec<Tensor2D>, y_real: &Vec<Tensor2D>) -> Scalar {
    
    // Assert both vectors have the same length
    assert_eq!(y_pred.len(), y_real.len());

    let mut loss: Scalar = Scalar::new(0.0);

    for (pred, real) in y_pred.iter().zip(y_real.iter()) {
        let loss_2d = (pred - real).pow(2.0);
        for row in 0..loss_2d.rows {
            for col in 0..loss_2d.cols {
                loss = &loss + &loss_2d.data[row][col];
            }
        }
    }

    loss
}
