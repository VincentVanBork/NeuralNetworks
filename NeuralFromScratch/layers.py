from functions import cross_entropy_loss_prime
from functions import cross_entropy_loss
from functions import ReLu, ReLu_prime, softmax, softmax_prime
import numpy as np

class Layer:
    def __init__(self, size:tuple, activation:tuple) -> None:
        super().__init__()
        self.weights = np.random.rand(size[1], size[0])
        self.input = np.zeros((1, size[0]))
        self.output:np.ndarray
        self.activation = activation[0]
        self.deactivation = activation[1]
        self.bias = np.ones((1, size[0]))
    def forward_propagation(self,inputs):
        self.input = inputs
        self.input = np.dot(self.input, self.weights)
        # print(self.output, "<-- out")
        # print(self.bias, "<-- bias")
        self.input += self.bias
        self.activate()

    def backward_propagation(self, error, learning_rate):
        print(error , "<----err", error.shape)
        print(self.weights , "<----weights", self.weights.shape)
        print(self.deriv() , "<----deriv", self.deriv().shape)
        print(self.output , "<----deriv", self.output.shape)

        self.weights -=  learning_rate * error* np.dot(np.dot(self.weights.T , self.deriv()), self.output)
        return error * np.dot(self.weights.T , self.deriv())
        
    def activate(self):
        self.output = self.activation(self.input)

    def deriv(self):
        return self.deactivation(self.output)


class Network:
    def __init__(self, epochs, learning_rate) -> None:
        super().__init__()
        self.layers:list[Layer] = []
        self.error:np.array
        self.loss = ...
        self.loss_prime = ...
        self.learning_rate = learning_rate

    def train(self, input_data, output_data):

        for data in zip(input_data, output_data):
            #might be faster if 0th elemtn is done outside instead of if
            for index, layer in enumerate(self.layers):
                if index != 0:
                    layer.forward_propagation(self.layers[index-1].output)
                else:
                    layer.forward_propagation(np.array([data[0]]))
            #true false loss
            self.error = self.loss(data[1], self.layers[-1].output)
            self.gradient_error = self.loss_prime(data[1], self.layers[-1].output)

            # print(self.error)
            # print(self.gradient_error)

            for index, layer in enumerate(reversed(self.layers)):
                print(index, '____INDEX____')
                if index != 0:
                    self.error = layer.backward_propagation(self.gradient_error, self.learning_rate)
                else:
                    layer.weights -= self.learning_rate * np.dot(self.gradient_error,layer.output.T)
                    self.error = np.dot(self.gradient_error, layer.output.T)
            
            # print(self.error)

    def add(self, layer:Layer):
        self.layers.append(layer)

    def use(self, loss_function, deriv_loss):
        self.loss = loss_function
        self.loss_prime = deriv_loss


n = Network(1, 1)
n.use(cross_entropy_loss, cross_entropy_loss_prime)
n.add(Layer((7,4),(ReLu, ReLu_prime)))
n.add(Layer((5,7),(ReLu, ReLu_prime)))
n.add(Layer((3,5),(softmax, softmax_prime)))

input_data = np.array([[5.1,3.5,1.4,0.2], [6.8,3.2,5.9,2.3]])
output_data = np.array([[1,0,0], [0,0,1]])
# print(zip(input_data, output_data))
n.train(input_data=input_data, output_data=output_data)