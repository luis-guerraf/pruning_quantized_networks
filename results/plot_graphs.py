# libraries
import matplotlib.pyplot as plt
import torch
import numpy as np
from numpy.random import uniform as rand
import colorsys

# Get data
metric = 'angles_kernel'
set = 'validation'
dataset = 'imagenet'
model = 'resnet_binary'
method = 'kernel'

fig = plt.figure()
ax = plt.subplot(111)
for ranking in ['large']:
    path = './paper/' + method + '/' + metric + '/' + set + '/' + dataset + '/' + model + '/' + ranking + '/Pruning_Accuracy.pt'
    data = torch.load(path)
    data = data[:, torch.linspace(0, 90, 10).type(torch.long)].numpy()

    # Plot
    for i in range(0, data.shape[0]):
        if i%2 != 0:
            continue

        hue = 1.0*i/data.shape[0]
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)

        # This can be commented out
        ax.plot(range(0, 100, 10), data[i, :], linestyle='-', marker='o', color=rgb, label=str('conv '+str(i)))

    if ranking == 'small':
        rgb = (0, 1, 0)
        label = 'Prune by smallest'
    elif ranking == 'random':
        rgb = (0, 0, 1)
        label = 'Prune randomly'
    elif ranking == 'large':
        rgb = (1, 0, 0)
        label = 'Prune by largest'

    # This can be commented out
    # plt.plot(range(0, 100, 10), np.mean(data, 0), linestyle='-', marker='o', color=rgb, label=label)

plt.xlabel('Filters pruned (%)')
plt.ylabel('Test accuracy')
# plt.title(dataset + ',' + model + ',' + ranking + method +  metric)
plt.title('Imagenet, Resnet-18, pruned kernels by smallest angles')

# Draw legend only
# chartBox = ax.get_position()
# ax.axis('off')
# ax.set_position([chartBox.x0+0.1, chartBox.y0+0.9, 0, 0])
# ax.legend()

plt.show()
