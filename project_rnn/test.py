import numpy as np
import matplotlib.pyplot as plt
import torch

a = torch.tensor([[1,2,3],[2,3,4]])
print(a.max(1)[0])