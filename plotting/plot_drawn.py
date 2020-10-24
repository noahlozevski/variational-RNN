import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plot

data_path = '../datasets/Task2/'
data_file = 'U25S25.TXT'

sig_data = genfromtxt(data_path + data_file, delimiter=' ', skip_header=1, dtype=int)

sig_matrix = np.zeros((10000, 10000), dtype=int)

for row in sig_data:
    sig_matrix[row[0]-25:row[0]+25, row[1]-25:row[1]+25] = 1
    if row[3]:
        xdelta = row[0] - xprev
        ydelta = row[1] - yprev
        delta = max(abs(xdelta), abs(ydelta))
        for i in range(1, delta):
            xstep = int(i*xdelta/delta)
            ystep = int(i*ydelta/delta)
            sig_matrix[xprev+xstep-25:xprev+xstep+25, yprev+ystep-25:yprev+ystep+25] = 1
    xprev = row[0]
    yprev = row[1]

plot.imshow(sig_matrix)
plot.show()
