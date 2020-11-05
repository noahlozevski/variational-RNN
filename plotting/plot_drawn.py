import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plot

data_path = '../datasets/Task1/'
data_file = 'U40S1.TXT'

sig_data = genfromtxt(data_path + data_file, delimiter=' ', skip_header=1, dtype=int)
print(len(sig_data))

sig_matrix = np.zeros((12000, 12000), dtype=int)

for row in sig_data:
    sig_matrix[12000-row[1]-25:12000-row[1]+25, row[0]-25:row[0]+25] = 1
    if row[3]:
        xdelta = 12000 - row[1] - xprev
        ydelta = row[0] - yprev
        delta = max(abs(xdelta), abs(ydelta))
        for i in range(1, delta):
            xstep = int(i*xdelta/delta)
            ystep = int(i*ydelta/delta)
            sig_matrix[xprev+xstep-25:xprev+xstep+25, yprev+ystep-25:yprev+ystep+25] = 1
    xprev = 12000 - row[1]
    yprev = row[0]

plot.imshow(sig_matrix)
plot.show()
