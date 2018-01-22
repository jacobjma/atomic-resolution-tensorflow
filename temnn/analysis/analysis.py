import numpy as np
from pyqstem.util import project_positions

def create_truth(entry, edge):
    
    positions=entry.model.get_positions()/sampling
    
    truth=MarkerCollection(truth[:,:2])
    #template_matching(truth,'hexagonal',strain=True,match_partial=False,scale=1.42,tol=.05)

    #labels=np.array(['hexagonal']*truth.num_markers)
    #labels[np.isnan(truth.get_property('rmsd'))]='other'

    #truth.set_labels(labels)
    
    #limits=[edge,size[0]-edge,edge,size[1]-edge]
    #truth = label_edges(truth,limits)
    
    return truth