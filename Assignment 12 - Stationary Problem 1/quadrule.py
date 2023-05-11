import numpy as np

class QuadratureRule:
  def __init__(self, order=1):
    if order == 1:
      self.weights = np.array([0.5])
      self.points = np.array([[1.0/3.0,1.0/3.0]])
    else:
      self.weights = np.array([1.0/6.0,1.0/6.0,1.0/6.0])
      self.points = np.array([[1.0/6.0,1.0/6.0],[2.0/3.0,1.0/6.0],[1.0/6.0,2.0/3.0]])     
      if order > 2: 
        print("Requested order " + str(order) + " is not implemented. Returning order 2 quadrature.")


  def getPointsAndWeights(self):
    return self.points,self.weights
