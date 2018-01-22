import numpy as np
from scipy.spatial.distance import cdist
from ase import io
from temnn.net import mods
import pyqstem.wave
from pyqstem.util import project_positions
import matplotlib.pyplot as plt
from temnn.net.labels import create_label

def to_tensors(batch):
    images = np.concatenate([b.image for b in batch],axis=0)
    labels = np.concatenate([b.label for b in batch],axis=0)
    
    return images, labels
    
class DataEntry(object):
    
    def __init__(self,wave=None,image=None,label=None,points=None,model=None):
        
        self.wave_file = wave
        self.image_file = image
        self.label_file = label
        self.model_file = model
        self.points_file = points
        
        self.reset()
    
    def load(self):
        
        if self.wave_file is not None:
            self._wave = pyqstem.wave.load(self.wave_file)
        
        if self.image_file is not None:
            self._image = np.load(self.image_file)
            if len(self._image.shape)==3:
                self._image = self._image.reshape((1,)+self._image.shape)
            
        if self.label_file is not None:
            self._label = np.load(self.label_file)
            if len(self._label.shape)==3:
                self._label = self._label.reshape((1,)+self._label.shape)
        
        if self.model_file is not None:
            self._model = io.read(self.model_file)
        
        if self.points_file is not None:
            npzfile = np.load(self.points_file)
            self._sites = npzfile['sites'] 
            self._classes = npzfile['classes'] 
        
    def create_image(self, ctf, sampling, blur, dose, MTF_param=None, concatenate=False):
        image = self._wave.apply_ctf(ctf).detect(resample=sampling,blur=blur,dose=dose,MTF_param=MTF_param)
        image = image.reshape((1,)+image.shape+(1,)).astype(np.float32)
        
        if concatenate:
            self._image = np.concatenate((self._image,image),axis=3)
        else:
            self._image = image
    
    def create_label(self,sampling,shape=None,width=12,num_classes=None):
        
        if shape is None:
            shape=self._image.shape[1:-1]
        
        self._label=create_label(self._sites[:,:2]/sampling,shape,width=width,classes=self._classes,null_class=True,num_classes=num_classes)
        #self._label=create_label(self._sites[:,:2]/sampling,shape,width=width,classes=None,null_class=True,num_classes=num_classes)
        #self._label=create_label(self._sites[:,:2]/sampling,shape,width=width,classes=None,null_class=False,num_classes=num_classes)
        self._label=self._label.reshape((1,)+self._label.shape)
    
    def create_sites(self,sampling,radius):
        
        self._sites = project_positions(self._model,radius)/sampling
    
    def pad(self,size):
        
        self._image=np.pad(self._image,((0,0),(0,size[0]-self._image.shape[1]),
                            (0,size[1]-self._image.shape[2]),(0,0)),'constant', constant_values=0)
        self._label=np.pad(self._label,((0,0),(0,size[0]-self._label.shape[1]),
                            (0,size[1]-self._label.shape[2]),(0,0)),'constant', constant_values=0)

    def crop(self, amount, sampling=None):
        if self._image is not None:
            self._image = self._image[:,amount:-amount,amount:-amount,:]
        if self._label is not None:
            self._label = self._label[:,amount:-amount,amount:-amount,:]
        if self._sites is not None:
            self._sites = self._sites - amount/sampling
            
    def local_normalize(self,sigma1,sigma2):
        if self._image is not None:
            self._image = mods.local_normalize(self._image,sigma1,sigma2)
    
    def normalize(self):
        self._image = mods.normalize(self._image)
    
    def random_brightness(self,low,high):
        self._image=mods.random_brightness(self._image,low,high)
    
    def random_contrast(self,low,high):
        self._image=mods.random_contrast(self._image,low,high)
    
    def random_gamma(self,low,high):
        self._image=mods.random_gamma(self._image,low,high)
    
    def random_flip(self):
        if self._sites is not None:
            self._image,self._label,self._sites = mods.random_flip(self._image,self._label,self._sites)
        else:
            self._image,self._label = mods.random_flip(self._image,self._label)
    
    def random_crop(self,image_size,sampling=None):
        orig_size=self._image.shape[1:3]
    
        n=np.random.randint(0,orig_size[0]-image_size[0])
        m=np.random.randint(0,orig_size[1]-image_size[1])

        self._image=self._image[:,n:n+image_size[0],m:m+image_size[1],:]
        
        if self._label is not None:
            self._label=self._label[:,n:n+image_size[0],m:m+image_size[1],:]
        
        if self._sites is not None:
            self._sites[:,0]-=n*sampling
            self._sites[:,1]-=m*sampling
            
    
    def as_tensors(self,return_all=False):
        if return_all:
            return self._image,self._label,self._sites,self._classes
        else:
            return self._image,self._label
    
    def view(self,axes=None,show_positions=True):
        if axes is None:
            fig,axes = plt.subplots(1,2)
        
        if (self._positions is not None)&show_positions:
            axes[0].plot(self._positions[:,0],self._positions[:,1],'x')
            axes[1].plot(self._positions[:,0],self._positions[:,1],'x')
            
        axes[0].imshow(self._image[0,:,:,0].T,cmap='gray')
        axes[1].imshow(self._label[0,:,:,0].T,cmap='gray')
        
        plt.show()
    
    def reset(self):
        self._model = None
        self._wave = None
        self._label = None
        self._image = None
        self._sites = None
        self._classes = None
        
class DataSet(object):

    def __init__(self,entries=None):
        
        if entries is None:
            self._entries=[]
        else:
            self._entries=entries
        
        self._num_examples = len(self._entries)
        self._epochs_completed = 0
        self._index_in_epoch = 0
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
        
    @property
    def entries(self):
        return self._entries
        
    def reset(self):
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def append(self,entry=None):
        if entry is None:
            entry = DataEntry()
        self._entries.append(entry)
        self._num_examples+=1

    def remove(self,index):
        del self._entries[index]
        self._num_examples-=1
    
    def entry(self,index):
        return self._entries[index]

    def split(self,number):
        part1 = DataSet(self._entries[:-number])
        part2 = DataSet(self._entries[-number:])
        return part1, part2
        
    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        # Shuffle for the first epoch
        
        if self._epochs_completed == 0 and start == 0:
            self._perm = np.arange(self._num_examples)
            if shuffle:
                np.random.shuffle(self._perm)

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            batch_rest_part = [self._entries[i] for i in self._perm][start:self._num_examples]
            
            if shuffle:
                np.random.shuffle(self._perm)
            
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            
            batch_new_part = [self._entries[i] for i in self._perm][start:end]
            
            batch = batch_rest_part + batch_new_part
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
        
            batch = [self._entries[i] for i in self._perm][start:end]

        return batch