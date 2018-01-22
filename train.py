import os
from datetime import datetime
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow as tf
from pyqstem.imaging import CTF
from temnn.net import net
from temnn.net.dataset import DataSet, DataEntry
from temnn.net.labels import create_label

def load(data_dir):
    
    waves = glob(data_dir+"wave/wave_*.npz")
    points = glob(data_dir+"points/points_*.npz")

    entries = [DataEntry(wave=w, points=p) for w,p in zip(waves,points)]
    
    return DataSet(entries)

def show_examples(data, size, n=3):
    
    image,label=next_example(data,size)
    
    fig,axarr=plt.subplots(image.shape[-1]+1,n)
    
    for i in range(n):
        
        for j in range(image.shape[-1]):
            im = axarr[j,i].imshow(image[0,:,:,j], interpolation='nearest', cmap='gray')
            
            divider = make_axes_locatable(axarr[j,i])
            cax1 = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax = cax1)
        
        im = axarr[1,i].imshow(label[0,:,:,0], cmap='jet')
        
        divider = make_axes_locatable(axarr[-1,i])
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax = cax2)
        
        if i < n - 1:
            image, label = next_example(data, size)
    
    plt.tight_layout()
    plt.show()
    
def next_example(data,size):

    sampling = np.random.uniform(.084,.09)
    Cs = np.random.uniform(-28,-32) * 10**4
    defocus = np.random.uniform(80,100)
    focal_spread = np.random.uniform(50,60)
    
    aberrations={'a22' : 50, 
                'phi22' : np.random.rand() * 2 * np.pi,
                'a40' : 1.4 * 10**6}
    
    dose = 10**np.random.uniform(2,4)
    
    c1=np.random.uniform(.9,1)
    c2=np.random.uniform(0,.01)
    c3=np.random.uniform(.3,.4)
    c4=np.random.uniform(2.4,2.6)
    
    mtf_param=[c1,c2,c3,c4]
    
    blur = np.random.uniform(5,7)
    
    entry=data.next_batch(1)[0]
    
    entry.load()
    
    ctf=CTF(defocus=defocus,Cs=Cs,focal_spread=focal_spread,aberrations=aberrations)
    
    entry.create_image(ctf,sampling,blur,dose,mtf_param)
    
    entry.create_label(sampling, width = int(.4/sampling))
    
    entry.local_normalize(12./sampling, 12./sampling)
    
    entry.random_crop((424,) * 2, sampling)
    
    entry.random_brightness(-.1,.1)
    entry.random_contrast(.9,1.1)
    entry.random_gamma(.9,1.1)
    
    entry.random_flip()
    image,label=entry.as_tensors()
    entry.reset()
    
    return image,label
   
def summary_image(y,size):
    return tf.reshape(tf.cast(tf.argmax(y,axis=3),tf.float32),(1,)+size+(1,))
    
if __name__ == "__main__":
    
    data_dir = "data/clusters/"
    summary_dir = "summaries/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    graph_path = 'graphs/clusters.ckpt'
    new_graph_path = 'graphs/clusters.ckpt'
    
    data = load(data_dir)
    
    batch_size = 1
    image_size = (424,424) # spatial dimensions of input/output
    image_features = 1 # depth of input data
    num_classes = 1 # number of predicted class labels
    kernel_num = 32 # number at the first level
    num_epochs = 100 # number of training epochs
    restore = True # restore previous graph
    save_every = 100 # iterations between saves
    loss_type = 'mse' # mse or cross_entropy
    nonlinearity = 'sigmoid' # sigmoid or softmax
    weight_decay = 0.002 # weight decay scale
    
    num_iterations = num_epochs*data.num_examples//batch_size
    
    show_examples(data, image_size, n=4)
    
    x = tf.placeholder(tf.float32, shape=(batch_size,)+image_size+(image_features,))
    y_ = tf.placeholder(tf.float32, shape=(batch_size,)+image_size+(num_classes,))
    
    y = net.graph(x,kernel_num,output_features=num_classes)
    
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    
    if loss_type=='mse':
        y = tf.nn.sigmoid(y)
        loss = tf.losses.mean_squared_error(y_,y)+weight_decay*tf.reduce_sum(reg_losses)
    else:
        if nonlinearity == 'softmax':
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))+weight_decay*tf.reduce_sum(reg_losses)
        else:
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,logits=y))+weight_decay*tf.reduce_sum(reg_losses)
    
    training_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    
    tf.summary.image('input',x,1)
    
    if num_classes==1:
        tf.summary.image('label',y_)
        
        if loss_type=='mse':
            tf.summary.image('inference',y)
        else:
            tf.summary.image('inference',tf.sigmoid(y))
    else:
        tf.summary.image('label',summary_image(y_,image_size))
        
        if loss_type=='mse':
            tf.summary.image('inference',summary_image(y,image_size))
        else:
            if nonlinearity == 'softmax':
                tf.summary.image('inference',summary_image(tf.nn.softmax(y),image_size))
            else:    
                tf.summary.image('inference',summary_image(tf.sigmoid(y),image_size))
    
    tf.summary.scalar('loss',loss)
    merged = tf.summary.merge_all()
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(summary_dir,sess.graph)
        
        if restore:
            saver.restore(sess, graph_path)
        
        for i in range(num_iterations):
            
            image,label = next_example(data,image_size)
            
            summary, _, loss_val, inference = sess.run([merged, training_step, loss, y], feed_dict={x: image, y_: label})
            
            writer.add_summary(summary, i)
            
            if ((i%save_every==0)&(i>0))|(i==num_iterations-1):
                save_path = saver.save(sess, new_graph_path)
                print('Model saved in file: %s' % save_path)
            
            print("Epoch: {0}/{1}[{2}/{3}]".format(i//num_iterations,num_epochs,i,num_iterations))