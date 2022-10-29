import numpy as np
from tensorflow import keras
from multiprocessing import Process
import subprocess
import warnings
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Dense, Dropout, experimental
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential

def reconstruire(img1,K_n):
    ex,ey,ez=img1.shape
    A = np.zeros((ex,ey,1), dtype=np.bool)
    h=ex
    l=ey
    z=1
    
    #longueur
    L1 = [ i for i in range(0,l-255,255)]+[l-255]
    L2 = [ 256+i for i in range(0,l,255) if 256+i < l]+[l]

    #hauteur
    R1 = [ i for i in range(0,h-255,255)]+[h-255]
    R2 = [ 256+i for i in range(0,h,255) if 256+i < h]+[h]
  
    n = 0
    for h1,h2 in zip(R1,R2):
        for l1,l2 in zip(L1,L2):
            if A[h1:h2,l1:l2].shape == K_n[n].shape:
                A[h1:h2,l1:l2] = K_n[n]
            else:
                A[h1:h2,l1:l2] = np.zeros(A[h1:h2,l1:l2].shape, dtype=np.bool)
            n+=1
    return A

def dice_coefficient(y_true,y_pred):
    eps = 1e-6
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection =keras.backend.sum(y_true_f*y_pred_f)
    return (2. * intersection) / (keras.backend.sum(y_true_f * y_true_f) + keras.backend.sum(y_pred_f * y_pred_f) + eps)

def get_mosaic_predict(img):
    A = []
    h,l,z = img.shape
    #longueur
    L1 = [ i for i in range(0,l-255,255)]+[l-255]
    L2 = [ 256+i for i in range(0,l,255) if 256+i < l]+[l]

    #hauteur
    R1 = [ i for i in range(0,h-255,255)]+[h-255]
    R2 = [ 256+i for i in range(0,h,255) if 256+i < h]+[h]

    for h1,h2 in zip(R1,R2):
        for l1,l2 in zip(L1,L2):
            A.append(img[h1:h2,l1:l2])
    return A

def mount_nextcloud(frompath, topath):
    """
    Mount a NextCloud folder in your local machine or viceversa.
    Example of usage:
        mount_nextcloud('rshare:/data/images', 'my_local_image_path')
    Parameters
    ==========
    * frompath: str, pathlib.Path
        Source folder to be copied
    * topath: str, pathlib.Path
        Destination folder
    """
    command = ["rclone", "copy", f"{frompath}", f"{topath}"]
    result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = result.communicate()
    if error:
        warnings.warn(f"Error while mounting NextCloud: {error}")
    return output, error

def conv2d_block(input_tensor, n_filters, kernel_size=3):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               padding="same")(input_tensor) # padding="valid"
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), 
               padding="same")(x)
    x = Activation("relu")(x)
    return x

def get_unet(input_img, n_filters,kernel_size=3):
    # contracting path # encoder
    c1 = conv2d_block(input_img, n_filters=n_filters*4, kernel_size=3) #The first block of U-net
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = conv2d_block(p1, n_filters=n_filters*8, kernel_size=3)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = conv2d_block(p2, n_filters=n_filters*16, kernel_size=3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c10 = conv2d_block(p3, n_filters=n_filters*16, kernel_size=3)
    p10 = MaxPooling2D((2, 2)) (c10)

    c12 = conv2d_block(p10, n_filters=n_filters*16, kernel_size=3)
    p12 = MaxPooling2D((2, 2)) (c12)

    c14 = conv2d_block(p12, n_filters=n_filters*16, kernel_size=3)
    p14 = MaxPooling2D((2, 2)) (c14)

    c16 = conv2d_block(p14, n_filters=n_filters*16, kernel_size=3)
    p16 = MaxPooling2D((2, 2)) (c16)

    c4 = conv2d_block(p16, n_filters=n_filters*32, kernel_size=3)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*64, kernel_size=3) # last layer on encoding path 
    
    # expansive path # decoder
    u6 = Conv2DTranspose(n_filters*32, (3, 3), strides=(2, 2), padding='same') (c5) #upsampling included
    u6 = concatenate([u6, c4])
    c6 = conv2d_block(u6, n_filters=n_filters*32, kernel_size=3)

    u17 = Conv2DTranspose(n_filters*16, (3, 3), strides=(2, 2), padding='same') (c6)
    u17 = concatenate([u17, c16])
    c17 = conv2d_block(u17, n_filters=n_filters*16, kernel_size=3)

    u15 = Conv2DTranspose(n_filters*16, (3, 3), strides=(2, 2), padding='same') (c17)
    u15 = concatenate([u15, c14])
    c15 = conv2d_block(u15, n_filters=n_filters*16, kernel_size=3)

    u13 = Conv2DTranspose(n_filters*16, (3, 3), strides=(2, 2), padding='same') (c15)
    u13 = concatenate([u13, c12])
    c13 = conv2d_block(u13, n_filters=n_filters*16, kernel_size=3)

    u11 = Conv2DTranspose(n_filters*16, (3, 3), strides=(2, 2), padding='same') (c13)
    u11 = concatenate([u11, c10])
    c11 = conv2d_block(u11, n_filters=n_filters*16, kernel_size=3)

    u7 = Conv2DTranspose(n_filters*16, (3, 3), strides=(2, 2), padding='same') (c11)
    u7 = concatenate([u7, c3])
    c7 = conv2d_block(u7, n_filters=n_filters*16, kernel_size=3)

    u8 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = conv2d_block(u8, n_filters=n_filters*8, kernel_size=3)

    u9 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = conv2d_block(u9, n_filters=n_filters*4, kernel_size=3)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

def get_mosaic(img,train=True):
    A = []
    if len(img.shape)==3:
        h,l,z = img.shape
    else:
        h,l = img.shape
    #longueur
    L1 = [ i for i in range(0,l-255,255)]+[l-255]
    L2 = [ 256+i for i in range(0,l,255) if 256+i < l]+[l]

        #hauteur
    R1 = [ i for i in range(0,h-255,255)]+[h-255]
    R2 = [ 256+i for i in range(0,h,255) if 256+i < h]+[h]

    for h1,h2 in zip(R1,R2):
        for l1,l2 in zip(L1,L2):
            if train:
                A.append(img[h1:h2,l1:l2])
            else:
                temp1=img[h1:h2,l1:l2]
                mask = np.zeros((256, 256, 1), dtype=np.bool)
                for x in range(255):
                    for y in range(255):
                        mask[x,y,0]=temp1[x,y]
                A.append(mask)
    return A

def launch_tensorboard(port, logdir):
    subprocess.call(['tensorboard',
                     '--logdir', '{}'.format(logdir),
                     '--port', '{}'.format(port),
                     '--host', '0.0.0.0'])