import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from neural import ActivationLayer, FCLayer, Network
from functions import ReLu, ReLu_prime, cross_entropy_loss, mse, mse_prime, softmax, softmax_prime, tanh, tanh_prime

if __name__ == "__main__":
    data = loadtxt('bezdekIris.csv', delimiter=',', skiprows=1, usecols=(0,1,2,3))
    input_iris = data.reshape(-1,1,4)
    output_iris = np.zeros((150,1,3))
    
    output_iris[:51] = [[1,0,0]]
    output_iris[50:101] = [[0,1,0]]
    output_iris[100:] = [[0,0,1]]
 
    print(output_iris.shape)
    print(input_iris.shape)

    # x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
    # print(x_train.shape)
    # y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    net = Network()
    net.add(FCLayer(4, 8))
    net.add(ActivationLayer(ReLu, ReLu_prime))
    net.add(FCLayer(8, 5))
    net.add(ActivationLayer(ReLu, ReLu_prime))
    net.add(FCLayer(5, 3))
    net.add(ActivationLayer(softmax, softmax_prime))

    # train
    # net.use(mse,mse_prime)
    # net.use(mse, simple_error_prime)
    net.use(cross_entropy_loss, cross_entropy_loss)
    # net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)
    for i in range(2):
        net.fit([input_iris[i]], [output_iris[i]], epochs=1, learning_rate=0.01)
        net.fit([input_iris[50+i]], [output_iris[50+i]], epochs=1, learning_rate=0.01)
        net.fit([input_iris[100+i]], [output_iris[100+i]], epochs=1, learning_rate=0.01)

    # net.fit(input_iris, output_iris, epochs=10, learning_rate=0.0001)

    # net.fit(input_iris[51:70], output_iris[51:70], epochs=10, learning_rate=0.01)
    # net.fit(input_iris[120:], output_iris[120:], epochs=10, learning_rate=0.01)
    
    # test
    # out = net.predict(x_train)
    out = net.predict(input_iris)
    print(out)
    
    plt.figure()
    # plt.bar(["setosa", "versicolor", "virginica"],out[3].reshape(3,))
    # plt.figure()
    # plt.bar(["setosa", "versicolor", "virginica"],out[64].reshape(3,))
    # plt.figure()
    plt.bar(["setosa", "versicolor", "virginica"],out[-1].reshape(3,))
    plt.show()