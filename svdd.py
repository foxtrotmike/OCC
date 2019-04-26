"""
Author: Fayyaz Minhas, DCIS, PIEAS, PO Nilore, Islamabad, Pakistan
Email/web: http://faculty.pieas.edu.pk/fayyaz/

A barebones implementation of Support Vector Data Descriptors for Novelty Detection
Demonstrates: Representation, Evaluation and Optimization
as well as the concept of loss functions and automatic differentiation
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from sklearn.kernel_approximation import RBFSampler

#Let's generate some data
inputs = -1*np.array([[0.3,0.3],[0,1],[1,0],[1,1]])
targets = np.array([1,-1,-1,1])


inputs = 4+np.random.randn(100,2)
inputs = np.vstack((inputs,-4+np.random.randn(100,2)))

# Mapping to the space of RBF features
transformer = RBFSampler(gamma=0.1).fit(inputs)
def transform(inputs):
    return transformer.transform(inputs)#inputs #


device = torch.device('cpu')
# device = torch.device('cuda') # Uncomment this to run on GPU
x = torch.from_numpy(transform(inputs)).float()
N, D_in,D_out = x.shape[0], x.shape[1], 1

def svddLoss(y_pred):
    zero = torch.Tensor([0]) 
    return torch.mean(torch.max(zero, y_pred))

# Create random Tensors for weights; setting requires_grad=True means that we
# want to compute gradients for these Tensors during the backward pass.
#Note: we have added one additional weight (for bias)
wb = torch.randn(D_in+1, D_out, device=device, requires_grad=True)

#lossf = nn.MSELoss()#nn.L1Loss()
learning_rate = 1e-1
C = 10


optimizer = optim.Adam([wb], lr=learning_rate)
L = [] #history of losses
for t in range(1000):
  # Forward pass: compute predicted y using operations on Tensors. Since w1
  # has requires_grad=True, operations involving w1 will cause
  # PyTorch to build a computational graph, allowing automatic computation of
  # gradients. 
  """
  # REPRESENTATION
  """
  a = wb[1:]
  r = wb[0]
  y_pred = torch.norm(x-a.flatten(),dim=1)**2-r**2 #Implementing w'x+b

  """
  # EVALUATION
  """
  # Compute and print loss. Loss is a Tensor of shape (), and loss.item()
  # is a Python number giving its value.
  #loss = (y_pred - y).pow(2).mean() #loss = lossf(y_pred,y)
  #loss = hinge(y,y_pred)  
  loss = svddLoss(y_pred)
  obj = C*loss+r**2 #empirical loss + regularization
  L.append((loss.item(),obj.item())) #save for history
  """
  #OPTIMIZATION
  """
  # Use autograd to compute the backward pass. This call will compute the
  # gradient of loss with respect to all Tensors with requires_grad=True.
  # After this call w1.grad will be Tensors holding the gradient
  # of the loss with respect to w1.
  obj.backward()
  optimizer.step()
  optimizer.zero_grad()

def clf(inputs): 
  return np.linalg.norm((inputs)-wbn[1:].flatten(),axis=1)**2-wbn[0]**2

wbn = wb.detach().numpy()
plt.close("all")
plt.plot(L)
plt.grid(); plt.xlabel('Epochs'); plt.ylabel('value');plt.legend(['Loss','Objective'])
#print("Predictions: ",clf(transform(inputs)))
#print("Weights: ",wbn)
plt.figure()


from plotit import plotit
plotit(inputs,clf=clf, conts=[-1,0,1],transform = transform)