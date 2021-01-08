import numpy as np

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

def ReLu(x):
    # print(type(x))
    # print(x.shape)
    # print(x,x.shape)
    return np.maximum(0,x)

def ReLu_prime(x):
    x[x<=0] = 0.00001
    x[x>0] = 1
    return x
    # return np.where(x > 0, 1, 0)
def softmax_vec(x):
    # print("softmax real",x)
    nominator = np.exp(x-np.max(x))
    denominator = np.exp(x-np.max(x)).sum()
    return nominator/denominator

def softmax_vec_prime(x, err):
    # print("softmax real",x)
    # print("softmax err",err)
    return x-err

def softmax(sth, x):
    # print("softmax predicted ",sth)
    # print("softmax real",x)
    nominator = np.exp(x-np.max(x))
    denominator = np.exp(x-np.max(x)).sum()
    return nominator/denominator

def softmax_prime(true, predicted):
    """true and predicted"""
    predicted = predicted + 1e-8
    # weird = np.log(x)
    true_values = true.reshape((1,-1))
    val = -true_values*np.log(predicted)
    # print("BACK SOFTMAX", val)
    return val
    # print("SOFTMAEX", predicted)
    # indices = np.argmax(true, axis=1).astype(int)
    # predicted_probability = predicted[np.arange(len(predicted)), indices]
    # log_predictions = np.log(predicted_probability)
    # loss = -1.0 * np.sum(log_predictions) / len(log_predictions)
    # return loss
