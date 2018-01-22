import numpy as np
from ase.lattice.hexagonal import Graphite
from ase import Atom,Atoms
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial import Voronoi
import itertools

def sheet(box,rotation,edge_tol=0):
    
    atoms = Graphite(symbol='C', latticeconstant={'a':2.46,'c':6.70}, 
                     directions=[[1,-2,1,0],[2,0,-2,0],[0,0,0,1]], size=(1,1,1))
    
    del atoms[atoms.get_positions()[:,2]<2]
    
    diagonal=np.hypot(box[0],box[1])*2
    n = np.ceil(diagonal/atoms.get_cell()[0,0]).astype(int)
    m = np.ceil(diagonal/atoms.get_cell()[1,1]).astype(int)
    
    atoms*=(n,m,1)
    
    atoms.set_cell(box)
    atoms.rotate('z',rotation)    
    atoms.center()

    del atoms[atoms.get_positions()[:,0]<edge_tol]
    del atoms[atoms.get_positions()[:,0]>box[0]-edge_tol]
    del atoms[atoms.get_positions()[:,1]<edge_tol]
    del atoms[atoms.get_positions()[:,1]>box[1]-edge_tol]
    
    return atoms
    
def grains(box,centers):
    
    def in_hull(p, hull):
        from scipy.spatial import Delaunay
        if not isinstance(hull,Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p)>=0
    
    vor = Voronoi(centers)
    atoms=Atoms()
    for i,center in enumerate(centers):
        if (center[0]>0)&(center[1]>0)&(center[0]<box[0])&(center[1]<box[1]):
            region = vor.vertices[vor.regions[vor.point_region[i]]]
            new_atoms = sheet(box,np.random.rand()*2*np.pi,-np.max(box))
            new_atoms = new_atoms[in_hull(new_atoms.get_positions()[:,:2],region)]
            atoms+=new_atoms

    atoms.set_cell(box)
    atoms.set_pbc(1)
    atoms.wrap()
    
    return atoms

def strain(positions,direction,cell,power=-3,amplitude=10**3,N=(64,64)):

    def lookup_nearest(x0, y0, x, y, z):
        xi = np.abs(x-x0).argmin()
        yi = np.abs(y-y0).argmin()
        return z[yi,xi]

    noise=spectral_noise.power_law_noise(N,power)
    x=np.linspace(0,cell[0],N[0])
    y=np.linspace(0,cell[1],N[1])

    positions[:,direction]+=amplitude*np.array([lookup_nearest(p[0], p[1], x, y, noise) for p in positions]).T
    
    return positions