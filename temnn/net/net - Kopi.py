import numpy as np
from temnn.net import layers
import tensorflow as tf

def conv_res_conv(x,kernel_num,name="conv_res_conv"):
    with tf.variable_scope(name):
        x=layers.conv_layer(x,kernel_num,name="conv1")
        x=layers.res_block(x,kernel_num,name="res_block")
        return layers.conv_layer(x,kernel_num,name="conv2")

def graph(x, kernel_num, output_features):
    
    down1=conv_res_conv(x,kernel_num,name="down1")
    pool1=layers.pool_layer(down1,name="pool1")
    
    down2=conv_res_conv(pool1,kernel_num*2,name="down2")
    pool2=layers.pool_layer(down2,name="pool2")
    
    down3=conv_res_conv(pool2,kernel_num*4,name="down3")
    pool3=layers.pool_layer(down3,name="pool3")

    bridge=conv_res_conv(pool3,kernel_num*8,name="bridge")

    up3=layers.upsample_layer(bridge,down3.get_shape(),name="upsample3")
    up3=layers.skip(up3,down3)
    up3=conv_res_conv(up3,kernel_num*4,name="up3")
    
    up2=layers.upsample_layer(up3,down2.get_shape(),name="upsample2")
    up2=layers.skip(up2,down2)
    up2=conv_res_conv(up2,kernel_num*2,name="up2")
    
    up1=layers.upsample_layer(up2,down1.get_shape(),name="upsample1")
    up1=layers.skip(up1,down1)
    up1=conv_res_conv(up1,kernel_num,name="up1")
    
    inference = layers.score_layer(up1,output_features=output_features)
    
    return inference
