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


