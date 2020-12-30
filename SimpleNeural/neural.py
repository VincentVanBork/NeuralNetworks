import numpy as np

def ReLu(x):
    # print(type(x))
    # print(x.shape)
    # print(x,x.shape)
    return np.maximum(0,x)

def ReLu_prime(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x
    # return np.where(x > 0, 1, 0)


def softmax(sth, x):
    print("softmax predicted ",sth)
    print("softmax real",x)
    e_x = np.exp(x - np.max(x))
    val = e_x / e_x.sum(axis=0)
    return val

def softmax_prime(sth, x):
    """true and predicted"""
    print("TRUE SHAPE ", sth.shape, sth)
    print("PREDICTED SHAPE  ",x.shape,x )
    x = x + 1e-10
    # weird = np.log(x)
    # problem = -sth.reshape((1,-1))*np.log(x)
    return -sth.reshape((1,-1))*np.log(x)

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError


# inherit from base class Layer
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error


class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)

        weights_error = np.dot(self.input.T.reshape((-1,1)), output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            # print('epoch %d/%d   error=%f' % (i+1, epochs, err))
            print('epoch', epochs, "errpr", err)

if __name__ == "__main__":
    ...
    # training data
    # x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
    # y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    # # network
    # net = Network()
    # net.add(FCLayer(2, 3))
    # net.add(ActivationLayer(ReLu, ReLu_prime))
    # net.add(FCLayer(3, 1))
    # net.add(ActivationLayer(ReLu, ReLu_prime))

    # # train
    # net.use(softmax, softmax_prime)
    # net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

    # # test
    # out = net.predict(x_train)
    # print(out)