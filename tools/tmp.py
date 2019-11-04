import torch
import matplotlib

matplotlib.verbose = True
from matplotlib import pyplot as plt

optimizer = torch.optim.SGD([{'params': torch.ones(10), }], lr=1e-3, momentum=.9)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=.01, step_size_up=2000,
                                              # mode='exp_range',
                                              gamma=.7, scale_mode='cycle',
                                              scale_fn=lambda x: 0.7 ** x,
                                              )
lrs = []
ms = []
for e in range(5):
    for batch_idx in range(3000):
        scheduler.step()
        lr = scheduler.get_lr()
        lrs.append(lr)
        momentum = optimizer.param_groups[0]['momentum']
        ms.append(momentum)
plt.plot(lrs)
plt.show()
