import numpy as np
import matplotlib.pyplot as plt


class Grid():
    def __init__(self,xlow,xhigh,ylow,yhigh,Nx,Ny):
        self.xlow = xlow
        self.xhigh = xhigh
        self.ylow = ylow
        self.yhigh = yhigh
        self.Nx = Nx
        self.Ny = Ny 
        
        self.createGrid()

        # create the determinant and the Inverse of the Jacobian 
        self.adet = None
        self.invJac = None
        self.updateTrafoInformation()


    def createGrid(self):    
        x = np.linspace(self.xlow,self.xhigh,self.Nx)
        y = np.linspace(self.ylow,self.yhigh,self.Ny)    
            
        # meshgrid -> #nodes x 2
        self.points = np.zeros([self.Nx*self.Ny,2])
        self.points[:,0] = np.kron( np.ones(self.Ny) , x )   # Ny times copy of x
        self.points[:,1] = np.kron( y , np.ones(self.Nx)  )  # Nx y[0]  and  Nx y[1]  and ...  
    
        # 1 and 2 initial the first two triangles and copy it 
        self.cells = np.zeros([2*(self.Nx-1)*(self.Ny-1),3],dtype=int)
        self.cells[0::2] = [0,1,self.Nx]
        self.cells[1::2] = [self.Nx+1,self.Nx,1]
    
        # 3 row index shift
        shift = np.arange(0,(self.Nx-1)*(self.Ny-1))
        shift = np.kron(shift , np.ones(2,dtype=int)  )    
        self.cells = self.cells + shift[:,None]

        shift = np.arange(0,self.Ny-1)
        shift = np.kron( shift , np.ones(2*(self.Nx-1),dtype=int) )    
        self.cells = self.cells + shift[:,None]

        
    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis('equal')

        ax.scatter(self.points[:,0] , self.points[:,1])
        ax.triplot(self.points[:,0],self.points[:,1], self.cells)
    
        plt.show()   


    def isBoundaryPoint(self,p):
        return ((p[0] <= self.xlow+1e-6 or p[0] >= self.xhigh-1e-6) or (p[1] <= self.ylow + 1e-6 or p[1] >= self.yhigh - 1e-6))


    def updateTrafoInformation(self):   
        verts = self.points[self.cells]
    
        # det(A) = ad-bc
        # abs as np.abs
        det = np.multiply(verts[:,1,0]-verts[:,0,0],verts[:,2,1]-verts[:,0,1]) - np.multiply(verts[:,2,0]-verts[:,0,0],verts[:,1,1]-verts[:,0,1])

        # adjoint matrix transposed 
        firstCol = np.stack((verts[:,2,1]-verts[:,0,1],verts[:,0,0]-verts[:,2,0]),axis=1)
        secondCol = np.stack((verts[:,0,1]-verts[:,1,1],verts[:,1,0]-verts[:,0,0]),axis=1)
        adjMatT = np.stack((firstCol,secondCol),axis=2)
        
        # inverse transposed
        self.invJacT = np.divide(adjMatT,det[:,None,None])
        
        self.adet = np.abs(det)
        
        
    def evalReferenceMap(self,xHat):
        verts = self.points[self.cells]   

        # define reference matrix and vector
        trafoVec = verts[:,0,:]
        trafoMat = np.stack((verts[:,1,:]-verts[:,0,:],verts[:,2,:]-verts[:,0,:]),axis=2)    
    
        trafoX =  trafoMat[:,None,:,0]*xHat[None,:] + trafoMat[:,None,:,1]*xHat[None,:]  + trafoVec[:,None]                  
        return trafoX            
         
    def getDivisions(self):
        return self.Nx , self.Ny

    def getDeterminants(self):
        return self.adet

    def getInverseJacobians(self):
        return self.invJac

    def getPoints(self):
        return self.points

    def getInnerIndices(self):
        bounds_low = np.array([self.xlow,self.ylow])
        bounds_high = np.array([self.xhigh,self.yhigh])

        return np.where(np.minimum(np.min(np.abs(self.points-bounds_low),axis=1),np.min(np.abs(self.points-bounds_high),axis=1)) >1e-6)[0]

    def getBoundaryIndices(self, locator = (lambda x,y: True)):
        innerInd = self.getInnerIndices()
        allInd = np.arange(np.shape(self.points)[0])
        boundaryInd = np.setdiff1d(allInd,innerInd)
        return boundaryInd[locator(self.points[boundaryInd,0],self.points[boundaryInd,1])]
    
     
class ShishkinTypeGrid(Grid):
    def __init__(self,xlow,xhigh,ylow,yhigh,Nx,Ny,ax,ay):
        self.ax = ax
        self.ay = ay
        super().__init__(xlow,xhigh,ylow,yhigh,Nx,Ny)
        
        self.transform()
        
    def scalePointsToUnit(self):
        #scale to [0,1]**2
        self.points[:,0] -= self.xlow
        self.points[:,1] -= self.ylow
        self.points[:,0] /= self.xhigh-self.xlow
        self.points[:,1] /= self.yhigh-self.ylow

    def scalePointsBack(self):
        #scale back to Omega
        self.points[:,0] *= self.xhigh-self.xlow
        self.points[:,1] *= self.yhigh-self.ylow
        self.points[:,0] += self.xlow
        self.points[:,1] += self.ylow     
        
    def transform(self):        
        self.scalePointsToUnit()
                    
        alpha = [self.ax,self.ay]
        for i in [0,1]:
            
            if alpha[i]<=0 or alpha[i]>=1:
                raise ValueError
            
            ## transform the x coords
            ind1 = self.points[:,i]<0.5
            ind2 = self.points[:,i]>=0.5            
            # transform lower 0.5
            self.points[:,i][ind1] *= alpha[i] / 0.5 

            # transform greater 0.5
            self.points[:,i][ind2] -= 0.5 
            self.points[:,i][ind2] *= (1-alpha[i]) / 0.5 
            self.points[:,i][ind2] += alpha[i]         

        self.scalePointsBack()

      
class GridAdapter():
  def __init__(self,grid):
    self.indHangingNodes = []
    self.indHangingNodesShared = []
    self.markedCells=[]
    self.cellCenters=[]
    self.grid = grid

  def markCells(self,f):
    cellCenters = np.mean(self.grid.points[self.grid.cells],axis=1)
    self.markedCells = np.where(f(cellCenters[:,0],cellCenters[:,1]))[0]

  def refineMarkedCells(self):  
    
    for index in self.markedCells:
      cell = np.copy(self.grid.cells[index])
    
      points = self.grid.points[cell]
        
      newPoint = 0.5*(points[1]+points[2])
      dists = np.linalg.norm(self.grid.points-newPoint,axis=1) 
      if min(dists)<1e-6:
        # new point exists
        indNewPoint = np.where(dists<1e-6)[0][0]
        ind = np.where(self.indHangingNodes==indNewPoint)[0][0]
        self.indHangingNodes.remove(indNewPoint)
        self.indHangingNodesShared.pop(ind)
      else: 
        # insert the new point
        indNewPoint = len(self.grid.points)
        if not self.grid.isBoundaryPoint(newPoint):
          self.indHangingNodes.append(indNewPoint)
          self.indHangingNodesShared.append(self.grid.cells[index][[1,2]])
        #endif
      #endif
      self.grid.points = np.append( self.grid.points , newPoint[None,:] , axis=0)
        
        
      ### one cell
      d1 = np.sum((self.grid.points[cell[0]] - self.grid.points[cell[1]])**2)
      d2 = np.sum((self.grid.points[indNewPoint] - self.grid.points[cell[1]])**2)
      d3 = np.sum((self.grid.points[cell[0]] - self.grid.points[indNewPoint])**2)    
        
      if d1 > max(d2,d3):  
        self.grid.cells[index,:] = np.array([indNewPoint , cell[0] , cell[1]  ])
      elif d2 > max(d1,d3):
        self.grid.cells[index,:] = np.array([cell[0] , cell[1] , indNewPoint ]) 
      else:      
        self.grid.cells[index,:] = np.array([cell[1] , indNewPoint  , cell[0] ])     
      
    
      ### the other cell  
      d1 = np.sum((self.grid.points[cell[0]] - self.grid.points[cell[2]])**2)
      d2 = np.sum((self.grid.points[indNewPoint] - self.grid.points[cell[2]])**2)
      d3 = np.sum((self.grid.points[cell[0]] - self.grid.points[indNewPoint])**2)    
        
      if d1 > max(d2,d3):  
        self.grid.cells = np.append( self.grid.cells , np.array([[ indNewPoint , cell[2] , cell[0]  ]]), axis=0   ) # new 1 2    
      elif d2 > max(d1,d3):
        self.grid.cells = np.append( self.grid.cells , np.array([[ cell[0] , indNewPoint , cell[2]  ]]), axis=0   ) # new 1 2    
      else:      
        self.grid.cells = np.append( self.grid.cells , np.array([[ cell[2] , cell[0] , indNewPoint]]), axis=0   ) # new 1 2    
    
    #endfor   
    self.markedCells = []
  #endfunction

  def markHangingCells(self): 
    for shared in self.indHangingNodesShared:
      ind = np.where(np.sum(self.grid.cells == shared[0],axis=1)*np.sum(self.grid.cells == shared[1],axis=1))[0][0]
      self.markedCells.append(ind)
    #endfor
  #end function
    
  def refine(self,f,n=1):
    for i in range(n):
      self.markCells(f)
      self.refineMarkedCells()
      while len(self.indHangingNodes)>0:
        self.markHangingCells()
        self.refineMarkedCells()
      #endwhile
    #endfor
    self.grid.updateTrafoInformation()
  #endfunction


  def show(self):
    self.grid.show()













      
      
