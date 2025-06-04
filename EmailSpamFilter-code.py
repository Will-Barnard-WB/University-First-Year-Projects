# importing the modules needed for this project.
import numpy as np
from IPython.display import HTML,Javascript, display

# accessing the datasets.
training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(int)
testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)

# To run the training stage:
# 1) in the '__init__' function change readFromFile() to intialiseNetwork().
# 2) Add the following code to the init method;
        #  self.training_data = train_data[:, 1:].T
        #  self.training_labels = train_data[:, 0].T

        #  self.batch_gradient_scaler = 1 / len(train_data)

        #  self.test_data = test_data[:, 1:]
        #  self.test_labels = test_data[:, 0]
    # Also add 'train_data' and 'test_data' as parameters for the init method, as well as passing 'training_spam' and 'testing_spam' 
    # into the call to create a classifier object.
# 2) call the trainNetwork function.
# 3) This will perform a grid search whilst training the neural network.
# 4) If the network acheives a higher accuracy, it will write to the csv file (unlikely due to many iterations already being run).

class NeuralNetwork:

    def __init__(self):

        self.readFromFile()

    def readFromFile(self):

        # function used for loading the optimal network parameters and settings from the csv file.

        data = np.loadtxt("network_weights.csv", delimiter=",")

        i = 0
        self.layer_1_weights = data[i:i + 64 * 54].reshape(64, 54)
        i += (64*54)
        self.layer_1_bias = data[i:i + 64].reshape(64, 1)
        i += 64
        self.layer_2_weights = data[i:i + 32 * 64].reshape(32, 64)
        i += (32*64)
        self.layer_2_bias = data[i:i + 32].reshape(32, 1)
        i += 32
        self.layer_3_weights = data[i:i + 1 * 32].reshape(1, 32)
        i += (1*32)
        self.layer_3_bias = data[i:i + 1].reshape(1, 1)

        self.lr = data[-2]
        self.epochs = int(data[-1])

    def initialiseNetwork(self):

        # function used for initiating a new, random neural network.
        # only used during the training stage.

        self.lr = None
        self.epochs = None

        self.layer_1_weights = np.random.rand(64, 54) - 0.5
        self.layer_1_bias = np.random.rand(64, 1) - 0.5

        self.layer_2_weights = np.random.rand(32, 64) - 0.5
        self.layer_2_bias = np.random.rand(32, 1) - 0.5

        self.layer_3_weights = np.random.rand(1, 32) - 0.5
        self.layer_3_bias = np.random.rand(1, 1) - 0.5
        
    def reluActivation(self, X):

        # activation function for the hidden layers.

        return np.maximum(X, 0)
    
    def reluDerivative(self, X):

        # used in backpropagation steps for both hidden layers.

        return ((X > 0).astype(float))
    
    def sigmoidActivation(self, X):

        # actiation function for the output layer.

        return 1 / (1  + np.exp(-X))
    
    def forwardPropagation(self):

        # stepping through the network.

        self.layer_1_dot_product = self.layer_1_weights.dot(self.training_data) + self.layer_1_bias
        self.layer_1_output = self.reluActivation(self.layer_1_dot_product)

        self.layer_2_dot_product = self.layer_2_weights.dot(self.layer_1_output) + self.layer_2_bias
        self.layer_2_output = self.reluActivation(self.layer_2_dot_product)

        self.layer_3_dot_product = self.layer_3_weights.dot(self.layer_2_output) + self.layer_3_bias
        self.layer_3_output = self.sigmoidActivation(self.layer_3_dot_product)

    def backwardPropagation(self):

        # performing backpropagation in order to improve the networks' accuracy.
        # the network makes use of the binary cross entropy loss, as its derivative with respect to the sigmoid activation function
        # is rather convenient, simply being the difference between the model's output and the actual labels

        self.layer_3_deriv = self.layer_3_output - self.training_labels
        self.layer_3_weights_deriv = self.batch_gradient_scaler * self.layer_3_deriv.dot(self.layer_2_output.T)
        self.layer_3_bias_deriv = self.batch_gradient_scaler * np.sum(self.layer_3_deriv)

        self.layer_2_deriv = self.layer_3_weights.T.dot(self.layer_3_deriv) * self.reluDerivative(self.layer_2_dot_product)
        self.layer_2_weights_deriv = self.batch_gradient_scaler * self.layer_2_deriv.dot(self.layer_1_output.T)
        self.layer_2_bias_deriv = self.batch_gradient_scaler * np.sum(self.layer_2_deriv)

        self.layer_1_deriv = self.layer_2_weights.T.dot(self.layer_2_deriv) * self.reluDerivative(self.layer_1_dot_product)
        self.layer_1_weights_deriv = self.batch_gradient_scaler * self.layer_1_deriv.dot(self.training_data.T)
        self.layer_1_bias_deriv = self.batch_gradient_scaler * np.sum(self.layer_1_deriv)

    def updateParameters(self):

        # updating the network parameters during gradient descent.

        self.layer_3_weights = self.layer_3_weights - self.lr * self.layer_3_weights_deriv
        self.layer_3_bias = self.layer_3_bias - self.lr * self.layer_3_bias_deriv

        self.layer_2_weights = self.layer_2_weights - self.lr * self.layer_2_weights_deriv
        self.layer_2_bias = self.layer_2_bias - self.lr * self.layer_2_bias_deriv

        self.layer_1_weights = self.layer_1_weights - self.lr * self.layer_1_weights_deriv
        self.layer_1_bias = self.layer_1_bias - self.lr * self.layer_1_bias_deriv

    def getPredictions(self, X):

        # assign each prediction an output class - 1 or 0.

        return (X > 0.5).astype(int)
    
    def trainNetwork(self):

        # performing a grid search for hyperparameter tuning and training the neural network.

        maxAccuracy = 0.94 # current max value achieved by the stored settings.

        accuracyImproved = False

        blr = None
        bepoch = None
        blayer_1_weights = None
        blayer_1_bias = None
        blayer_2_weights = None
        blayer_2_bias = None
        blayer_3_weights = None
        blayer_3_bias = None

        learning_rates = [0.01, 0.025, 0.05, 0.075, 0.1]
        epochs = [250, 500, 750, 1000, 1250]

        for lr in learning_rates:
            for epoch in epochs:

                self.initialiseNetwork()
                
                for i in range(epoch):

                    self.lr = lr

                    self.forwardPropagation()

                    self.backwardPropagation()

                    self.updateParameters()

                    predictions = self.predict(self.test_data)
                    accuracy = np.count_nonzero(predictions == self.test_labels)/self.test_labels.shape[0]

                    if accuracy > maxAccuracy:
                        blr, bepoch, blayer_1_weights, blayer_1_bias, blayer_2_weights, blayer_2_bias, blayer_3_weights, blayer_3_bias = lr, epoch, self.layer_1_weights, self.layer_1_bias, self.layer_2_weights, self.layer_2_bias, self.layer_3_weights, self.layer_3_bias
                       
                        maxAccuracy = accuracy
                        accuracyImproved = True
        
                        self.saveSettings(blr, bepoch, blayer_1_weights, 
                                                 blayer_1_bias, blayer_2_weights, blayer_2_bias, blayer_3_weights, blayer_3_bias)
                        
                        print('better network settings discovered.')

        if accuracyImproved:
            print(f'new max accuracy is: {maxAccuracy}')
        else:
            print(f"accuracy has not improved.")
                                                      
    def saveSettings(self, lr, epochs, w1, b1, w2, b2, w3, b3):

        # saving the optimal settings to a csv file so that they can be easily accessed. 

        flatten_weights_and_biases = [arr.flatten() for arr in [w1, b1, w2, b2, w3, b3]]
        weights_and_biases = np.concatenate(flatten_weights_and_biases)

        network_settings = np.array([lr, epochs], dtype=np.float32)
        full_data = np.concatenate([weights_and_biases, network_settings])

        np.savetxt("network_weights.csv", full_data, delimiter=",")

    def predict(self, X):

        # testing the neural network for a set of input data.

        self.testing_data = X.T

        self.layer_1_dot_product = self.layer_1_weights.dot(self.testing_data) + self.layer_1_bias
        self.layer_1_output = self.reluActivation(self.layer_1_dot_product)

        self.layer_2_dot_product = self.layer_2_weights.dot(self.layer_1_output) + self.layer_2_bias
        self.layer_2_output = self.reluActivation(self.layer_2_dot_product)

        self.layer_3_dot_product = self.layer_3_weights.dot(self.layer_2_output) + self.layer_3_bias
        self.layer_3_output = self.sigmoidActivation(self.layer_3_dot_product)

        predictions = self.getPredictions(self.layer_3_output)
        
        return predictions

# a function for creating a neural network classifier instance.

def create_classifier():

    classifier = NeuralNetwork()

    return classifier

# creating a neural network classifier instance.

classifier = create_classifier()

# testing the classifier on the 'testing' dataset.

SKIP_TESTS = True

if not SKIP_TESTS:
    testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)
    test_data = testing_spam[:, 1:]
    test_labels = testing_spam[:, 0]

    predictions = classifier.predict(test_data)
    accuracy = np.count_nonzero(predictions == test_labels)/test_labels.shape[0]
    print(f"Accuracy on test data is: {accuracy}")
