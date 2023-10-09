# Deep-Learning

The code is an implementation of binary logistic regression from scratch. Let me explain what's happening step by step:

Data Generation: Generate synthetic data using the make_blobs function from scikit-learn. This data consists of two classes (centers=2) in a two-dimensional feature space.
Data Visualization: Create a scatter plot to visualize the generated data points, where points from different classes are colored differently.
Sigmoid Function: Define the sigmoid function, which is used to model the probability of an example belonging to class 1. The sigmoid function takes a linear combination of features (weighted sum) and squashes it into the range [0, 1].
Prediction Function: The predict function computes predictions for the entire dataset based on the current weights.
Loss Function (Binary Cross-Entropy): The loss function calculates the binary cross-entropy loss between the true labels (Y) and the predicted probabilities (Y_). It ensures numerical stability by clipping values close to 0 or 1.
Weight Update Function: The update function performs one iteration of weight updates using gradient descent. It computes the gradient of the loss with respect to the weights and updates the weights accordingly.
Training Function: The train function trains the logistic regression model. It initializes the weights, performs multiple epochs of training, and prints the loss at regular intervals.
Training the Model: Call the train function with your data (X and Y) and specify a learning rate. This trains the logistic regression model and updates the weights.
Plotting the Decision Boundary: After training, define a set of x1 values and calculate the corresponding x2 values based on the learned weights. These x1 and x2 values are used to plot the decision boundary of the logistic regression model.
Final Visualization: Create a scatter plot of the data points and overlay the decision boundary on the plot to visualize how the logistic regression model separates the two classes.

The code essentially implements a basic logistic regression model and demonstrates its application on synthetic data for binary classification. The model learns to classify points into one of two classes based on the features in the dataset.
