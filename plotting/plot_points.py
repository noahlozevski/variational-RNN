import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plot

data_path = '../datasets/Task2/'
data_file = 'U1S1.TXT'

sig_data = genfromtxt(data_path + data_file, delimiter=' ', skip_header=1, dtype=int)
print(len(sig_data))

sig_matrix = np.zeros((10000, 10000), dtype=int)

for row in sig_data:
    sig_matrix[row[0]-25:row[0]+25, row[1]-25:row[1]+25] = 1

plot.imshow(sig_matrix)
plot.show()
