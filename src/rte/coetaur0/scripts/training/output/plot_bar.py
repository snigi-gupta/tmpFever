#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

##### MAIN

save_path = "./bar_plot.png"

# data to plot
title = 'Accuracy by architecture'
n_groups = 3
dev_acc = (32, 88, 91)
test_acc = (31, 34, 36)
architectures = ('FEVER Baseline', 'ESIM + SA', 'ESIM + SA + AGG (3-layer)')
ylabel = 'Label Accuracy (%)'
left_col_label = 'Dev Set'
right_col_label = 'Test Set'

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.3
opacity = 0.5

rects1 = plt.bar(index, dev_acc, bar_width,
alpha=opacity,
color='b',
label=left_col_label)

rects2 = plt.bar(index + bar_width, test_acc, bar_width,
alpha=opacity,
color='g',
label=right_col_label)

#plt.xlabel('Architecture')
plt.ylabel(ylabel)
plt.title(title)
plt.xticks(index + bar_width, architectures)
plt.yticks(dev_acc)
plt.legend()

plt.tight_layout()
plt.show()
fig.savefig(save_path)


