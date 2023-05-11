import numpy as np
import matplotlib.pyplot as plt


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



class GlobalBasis():
    def __init__(self,grid,basis):
        self.grid = grid
        self.basis = basis
    
    def evalPhi(self,xHat,globalInd):
        supp,localInd = self.evalDOFMap(globalInd)
        
        valHat = self.basis.evalPhi(xHat)
        
        return valHat[:,localInd]     
    
    
    def evalDOFMap(self,globalInd):
        supp,localInd = np.where(self.grid.cells == globalInd)
        return supp,localInd     

    def evalSharedDOFMap(self, globalIndI , globalIndJ):
        supp_I,localIndices_I = self.evalDOFMap(globalIndI)
        supp_J,localIndices_J = self.evalDOFMap(globalIndJ)

        supp_IJ,tmpLocalIndices_I,tmpLocalIndices_J = np.intersect1d(supp_I,supp_J,assume_unique=False,return_indices=True)
        localIndices_I = localIndices_I[tmpLocalIndices_I]
        localIndices_J = localIndices_J[tmpLocalIndices_J]

        return supp_IJ , localIndices_I , localIndices_J
        
    
    




      
      
