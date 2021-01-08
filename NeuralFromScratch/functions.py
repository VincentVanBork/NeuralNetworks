import numpy as np
def ReLu(x):
    return np.where(x > 0, x, x * 0.5)

def ReLu_prime(x):
    return np.where(x > 0, 1,  0.5)


# loss function and its derivative
def mse(y_true, y_pred):
    #wartosc
    x = np.mean(np.power(y_true-y_pred, 2))
    # print(x)
    return x

def mse_prime(y_true, y_pred):
    #wektor
    x = 2*(y_pred-y_true)/y_true.size
    # print(x)
    return x


def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def softmax_prime(x,axis=None):
    return np.array([ 
            [
                np.array([
                    [
                        (i*(1-j)) if index_j == index_i else (-i*j) \
                        for index_j, j in enumerate(x[0])
                    ]
                ]).sum()\
                for index_i, i in enumerate(x[0]) 
            ] 
        ])
        

def cross_entropy_loss(true, predicted):
    predicted += 1e-8
    err =  true * -1*np.log(predicted)
    return err


def cross_entropy_loss_prime(true, predicted):
    return predicted - true

if __name__ == "__main__":
    ...
    # M = np.array([[1,0,0],[0,1,0],[0,0,1]])
    # x = np.array([[-0.09056757,  1.52774372,  1.82126822]])
    # # print(np.subtract(M,y))
    # for i in x[0]:
    #     print(np.array([[i*(1-j) if i==j else -i*j for j in x[0]]]).sum())
    
    # x:np.array = np.array([[0.276935,1.6428,1.5411]])
    # # print(x)
    # # print(M)
    # print(y[0][0])
    # print(y[0][1])
    # print(y[0][2])

    # sth = y[0][0]*(1-y[0][0])
    # sth2 = -  y[0][0] *  y[0][1]
    # sth3 = - y[0][0] *  y[0][2]
    # print(sth+sth+sth3)
    # print((y.dot(np.subtract(M,y))))
    # print()

    