import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import cdist

def project_positions(atoms,distance=1,return_counts=False):
    
    positions=atoms.get_positions()[:,:2]

    clusters = fcluster(linkage(positions), distance, criterion='distance')
    unique, indices = np.unique(clusters, return_index=True)
    positions = np.array([np.mean(positions[clusters==u] ,axis=0) for u in unique])
    counts = np.array([np.sum(clusters==u) for u in unique])
    
    if return_counts:
        return positions, counts
    else:
        return positions
    
def create_label(positions,shape,width,classes=None,null_class=False,num_classes=None):
    
    if classes is None:
        classes=[0]*len(positions)
    
    if num_classes is None:
        num_classes = np.max(classes)+1
    
    x,y=np.mgrid[0:shape[0],0:shape[1]]
    
    labels=np.zeros(shape+(num_classes,))

    for p,c in zip(positions,classes):
        p_round=np.round(p).astype(int)
        
        min_xi = np.max((p_round[0]-width*4,0))
        max_xi = np.min((p_round[0]+width*4+1,shape[0]))
        min_yi = np.max((p_round[1]-width*4,0))
        max_yi = np.min((p_round[1]+width*4+1,shape[1]))
        
        xi = x[min_xi:max_xi,min_yi:max_yi]
        yi = y[min_xi:max_xi,min_yi:max_yi]
        v=np.array([xi.ravel(),yi.ravel()])
        
        labels[xi,yi,c]+=np.exp(-cdist([p],v.T)**2/(2*width**2)).reshape(xi.shape)

    if null_class:
        labels=np.concatenate((labels,1-np.sum(labels,axis=2).reshape(labels.shape[:2]+(1,))),axis=2)
    
    return labels