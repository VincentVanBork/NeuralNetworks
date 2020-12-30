import numpy as np
from neural import softmax_vec
a = np.array([[2,-0.000000015,0.000000001]])
b = np.array([[1,0,3]])
c = ([[-0.59064021, -0.38183184, -0.94776724]])
# print(-a*np.log(b))

# print(np.exp(a-max(a)))
# print(np.max(a))
# print(np.log(c))
print(softmax_vec([0.7105232914556173, 0.42941813633183856, 0.5461254116807236]))