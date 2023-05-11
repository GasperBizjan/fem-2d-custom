import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as triang


class StationaryProblem():
  def __init__(self, globalBasis, quadrature, dirichletLocations = (lambda x,y: True), dirichletValues = (lambda x,y: 0.0)):

    self.grid = globalBasis.grid
    self.basis = globalBasis.basis

    self.globalBasis = globalBasis

    self.quadrature = quadrature
        
    # Boundary Conditions
    self.dirichletDOFs = self.grid.getBoundaryIndices(dirichletLocations)
    self.allDOFs = np.arange(np.shape(self.grid.points)[0])
    self.freeDOFs = np.setdiff1d(np.arange(np.shape(self.grid.points)[0]),self.dirichletDOFs)
    
    # Precompute data that is needed all the time.
    # x_k and w_k
    self.xkHat,self.wkHat = self.quadrature.getPointsAndWeights()
    # F_T(x_k)
    self.xkTrafo = self.grid.evalReferenceMap(self.xkHat)
    # DF_T -> det and inverse
    self.dSh = self.grid.getDeterminants()
    
    self.invJacT = self.grid.getInverseJacobiansT()
    
    # phi(x_k) and grad_phi(x_k)
    self.phi = self.basis.evalPhi(self.xkHat)
    self.gradPhi = self.basis.evalGradPhi(self.xkHat)

    # Initialize
    self.systemMatrix = np.zeros((np.shape(self.grid.points)[0],np.shape(self.grid.points)[0]))
    self.systemVector = np.zeros(np.shape(self.grid.points)[0])
    self.solution = None

    self.mass = np.zeros((np.shape(self.grid.points)[0],np.shape(self.grid.points)[0]))
    self.diffusion = np.zeros((np.shape(self.grid.points)[0],np.shape(self.grid.points)[0]))

    self.initialMass()
    self.initialDiffusion()
    self.a = 0.0
    self.c = 0.0
    self.assembleBoundaryConditions(dirichletValues)


  def assembleBoundaryConditions(self, dirichletValues):
    self.systemMatrix[self.dirichletDOFs,self.dirichletDOFs] = 1.0
    self.systemVector[self.dirichletDOFs] = dirichletValues(self.grid.points[self.dirichletDOFs,0],self.grid.points[self.dirichletDOFs,1])
  
  def addSource(self, f):
    for i in self.freeDOFs:
      supp,localInd = self.globalBasis.evalDOFMap(i)
      for T,loc_i in zip(supp,localInd):
        self.systemVector[i] += self.dSh[T] * np.sum(self.phi[:,loc_i] * self.wkHat * f(self.xkTrafo[T,:,0],self.xkTrafo[T,:,1]))

  def addDiscreteSource(self,vec):
    self.systemVector += self.mass.dot(vec)

  def initialDiffusion(self):
    for i in self.freeDOFs:
      for j in self.allDOFs:

        supp_IJ,localIndices_I,localIndices_J = self.globalBasis.evalSharedDOFMap(i,j)

        for T,loc_i,loc_j in zip(supp_IJ,localIndices_I,localIndices_J):
          for k in range(self.xkHat.shape[0]):
            self.diffusion[i,j] += self.dSh[T] * self.wkHat[k] * np.dot(np.dot(self.invJacT[T,:,:],self.gradPhi[k,loc_j,:]),np.dot(self.invJacT[T,:,:],self.gradPhi[k,loc_i,:]))


  def initialMass(self):
    for i in self.freeDOFs:
      for j in self.allDOFs:

        supp_IJ,localIndices_I,localIndices_J = self.globalBasis.evalSharedDOFMap(i,j)

        for T,loc_i,loc_j in zip(supp_IJ,localIndices_I,localIndices_J):
          for k in range(self.xkHat.shape[0]):
            self.mass[i,j] +=  self.dSh[T] * self.wkHat[k] * self.phi[k,loc_j] * self.phi[k,loc_i]

  def setReaction(self,c):
    self.c = c

  def setDiffusion(self,a):    
    self.a = a
    
  def setSolution(self,vec):
     self.solution = vec
    
  def getSolution(self):
     return self.solution     
    
  def assemble(self):
    self.systemMatrix += self.c*self.mass + self.a*self.diffusion

  def resetSystemVector(self):
    self.systemVector[:] = 0
  
  def resetSystemMatrix(self):
    self.systemMatrix[:,:] = 0

  def solve(self):
    self.solution = np.linalg.solve(self.systemMatrix , self.systemVector)

  def show(self):
    fig = plt.figure()
    # limits
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim([self.globalBasis.grid.xlow, self.globalBasis.grid.xhigh])
    ax.set_ylim([self.globalBasis.grid.ylow, self.globalBasis.grid.yhigh])
    ax.axis('equal')
    Triangulation = triang.Triangulation(self.globalBasis.grid.points[:,0],self.globalBasis.grid.points[:,1],self.globalBasis.grid.cells)
    tri = ax.tripcolor(Triangulation,self.solution,shading='gouraud')
    fig.colorbar(tri)
    plt.show()














