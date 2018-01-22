import numpy as np
from skimage.filters import gaussian
from scipy.ndimage.filters import gaussian_filter

def normalize(images, epsilon=1e-12):
    
    if len(images.shape)==4:
        for i in range(images.shape[0]):
            for j in range(images.shape[3]):
                images[i,:,:,j] = images[i,:,:,j] - np.mean(images[i,:,:,j])
                images[i,:,:,j] = images[i,:,:,j] / np.sqrt(np.std(images[i,:,:,j])**2 + epsilon)
    else:
        images=(images-np.mean(images))/np.sqrt(np.std(images)**2 + epsilon)
    
    return images

def local_normalize(images, sigma1, sigma2):
    if len(images.shape)==4:
        for i in range(images.shape[0]):
            
            B=np.zeros_like(images[0,:,:,0])
            S=np.zeros_like(images[0,:,:,0])
            for j in range(images.shape[3]):
                B+=gaussian_filter(images[i,:,:,j],sigma1)
            
            for j in range(images.shape[3]):
                images[i,:,:,j] = images[i,:,:,j] - B/images.shape[3]
            
            for j in range(images.shape[3]):
                S+=np.sqrt(gaussian_filter(images[i,:,:,j]**2, sigma2))
            
            for j in range(images.shape[3]):
                images[i,:,:,j] = images[i,:,:,j] / (S/images.shape[3])
    else:
        images = (images-np.min(images))/(np.max(images)-np.min(images))
        images = images - gaussian(images,sigma1)
        images = images / np.sqrt(gaussian(images**2, sigma2))
        
    return images

def random_flip(images,labels,positions=None):
    
    for i in range(len(images)):
        if np.random.rand() < .5:
            images[i,:,:,:] = np.fliplr(images[i,:,:,:])
            labels[i,:,:,:] = np.fliplr(labels[i,:,:,:])
            
            if positions is not None:
                positions[:,1]-=images.shape[1]/2
                positions[:,1]*=-1
                positions[:,1]+=images.shape[1]/2-1
        if np.random.rand() < .5:
            images[i,:,:,:] = np.flipud(images[i,:,:,:])
            labels[i,:,:,:] = np.flipud(labels[i,:,:,:])
            
            if positions is not None:
                positions[:,0]-=images.shape[1]/2
                positions[:,0]*=-1
                positions[:,0]+=images.shape[1]/2-1
    
    if positions is not None:
        return images,labels,positions
    else:
        return images,labels

def random_crop(images,labels,size,positions=None):
    
    orig_size=images.shape[1:3]
    
    if orig_size>size:
        
        n=np.random.randint(0,orig_size[0]-size[0])
        m=np.random.randint(0,orig_size[1]-size[1])
    
        images=images[:,n:n+size[0],m:m+size[1],:]
        labels=labels[:,n:n+size[0],m:m+size[1],:]
        
        if positions is not None:
            positions[:,0]-=n
            positions[:,1]-=m
    
    if positions is not None:
        return images, labels, positions
    else:
        return images, labels

def random_blur(images,low,high=None):
    if high is None:
        high=low
    for i in range(len(images)):
        sigma=np.random.uniform(low,high)
        images[i,:,:,0]=gaussian(images[i,:,:,0].astype(float),sigma)
    return images

def random_brightness(images,low,high=None):
    if high is None:
        high=low
    for i in range(len(images)):
        images[i,:,:,0]=images[i,:,:,0]+np.random.uniform(low,high)
    return images

def random_contrast(images,low,high=None):
    if high is None:
        high=low
    for i in range(len(images)):
        mean=np.mean(images[i,:,:,0])
        images[i,:,:,0]=(images[i,:,:,0]-mean)*np.random.uniform(low,high)+mean
    return images
    
def random_gamma(images,low,high=None):
    if high is None:
        high=low
    for i in range(len(images)):
        min=np.min(images[i,:,:,0])
        images[i,:,:,0]=(images[i,:,:,0]-min)*np.random.uniform(low,high)+min
    return images