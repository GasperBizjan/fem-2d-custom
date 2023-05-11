import numpy as np

class Basis():
  #def __init__(self):
  #  pass
  
  def evalPhi(self,xHat):
    N = np.shape(xHat)[0]
    phi = np.zeros((N,3))
    phi[:,0] = 1 - xHat[:,0] - xHat[:,1]
    phi[:,1] = xHat[:,0]
    phi[:,2] = xHat[:,1]
    return phi    
  
  def evalGradPhi(self,xHat):
    N = np.shape(xHat)[0]
    gradPhi = np.zeros((N,3,2))
    gradPhi[:,0,0] = -1
    gradPhi[:,0,1] = -1
    gradPhi[:,1,0] = 1  
    gradPhi[:,1,1] = 0
    gradPhi[:,2,0] = 0  
    gradPhi[:,2,1] = 1   
    return gradPhi    
      
  def show(self, uHat):
      
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    
    xHat = np.array([[0,0],[0,1.0],[1.0,0]])
    phi = self.evalPhi(xHat)

    ax.plot_trisurf(xHat[:,0],xHat[:,1],  phi.dot(uHat) )
    plt.show()

