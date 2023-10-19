# Deep-Learning

# Binary-Classification
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

# Multi-layer Perceptron

The code is implementing a simple Artificial Neural Network (ANN) to classify the Titanic dataset. Here's a step-by-step explanation of what's happening:

Importing libraries: random, pandas, numpy, matplotlib.pyplot, and seaborn libraries are imported for various tasks. The line %matplotlib inline is used in Jupyter Notebook to display plots in the notebook.

Reading the Titanic dataset: The Titanic dataset is read from a CSV file into a Pandas DataFrame.

Data preprocessing: Two dictionaries (dict_live and dict_sex) are defined to map values in the dataset. A new column 'Bsex' is created by applying a lambda function to map the 'Sex' column values to binary values (0 for 'male' and 1 for 'female'). The features (Pclass and Bsex) and labels (Survived) are extracted from the dataset.

Train-test split: The dataset is split into training and testing sets using train_test_split from scikit-learn.

Activation functions: Two activation functions, sigmoid and ReLU (Rectified Linear Unit), are defined. The ReLU function is used as an activation function for hidden layers in the neural network.

Neural network training: The train_ANN function is defined to train the neural network. It initializes random weights and biases for two hidden layers and an output layer.
It then iterates through the training data to perform the following steps:
Feedforward pass: Compute the outputs of each layer using the ReLU activation function for hidden layers and the sigmoid activation function for the output layer.
Backpropagation: Calculate the deltas (errors) at each layer to update the weights and biases using gradient descent.
Compute and record the loss for each training instance.
The average loss for each batch of 60 training instances is computed and plotted in a graph.

Making predictions: The ANNPredict function is defined to make predictions on the test set using the trained weights and biases. It also uses the ReLU activation function for hidden layers and the sigmoid function for the output layer.

Model evaluation: The code calculates a confusion matrix using sklearn.metrics.confusion_matrix to evaluate the performance of the model on the test data.
It also displays a heatmap of the confusion matrix using seaborn to visualize the performance of the ANN.

In summary, this code trains a simple two-hidden-layer ANN to predict whether passengers on the Titanic survived or not based on their class and gender. The model's performance is evaluated using a confusion matrix.
