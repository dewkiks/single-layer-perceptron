# Single Layer Perceptron
A single layer neural network that is purely created using numpy which can be used to predict 'AND' gate and 'OR' gate by training it with their inputs and outputs
# Working
* Initialization (__init__): Randomly initializes the weights and bias for the model. Sets the learning rate for updating parameters during training.
* Forward Pass (forward_pass): Computes the dot product of input data and weights. Adds the bias and applies the sigmoid activation to get predictions.
* Loss Calculation (compute_loss): Calculates the Mean Squared Error (MSE) between predicted and actual outputs to evaluate the model's performance.
* Backpropagation (back_propogation): Computes the gradient of the loss with respect to weights and bias using the chain rule. Prepares gradients for weight and bias updates. Parameter Update (update_parameters): Updates weights and bias by subtracting the product of gradients and learning rate (gradient descent).
* Training (training): Repeats the forward pass, loss calculation, backpropagation, and parameter update for a specified number of epochs. Prints loss every 100 epochs to track training progress.
* Prediction: After training, the model can predict the AND output for new input values. This is a simple neural network to model the AND operation using gradient descent.
# Video Demonstration
https://github.com/user-attachments/assets/044acf60-53fe-4a4b-8c0b-0784092896ed

