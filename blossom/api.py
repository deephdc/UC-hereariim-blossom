from functools import wraps
from genericpath import isfile
import shutil
import tempfile
from marshmallow import missing
import yaml
from webargs import fields

#127.0.0.1

from zipfile import ZipFile
from skimage.io import imread, imsave, imread_collection, concatenate_images
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
from tensorflow import keras
from focal_loss import BinaryFocalLoss

# from tensorflow.keras import backend as K
import numpy as np
from skimage.transform import resize

from PIL import Image
from aiohttp import web #HTTPBadRequest
from webargs import fields, validate
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import blossom.config as cfg
import blossom.path as paths
from keras import backend
import sys
import random
import warnings

import numpy as np
import imageio
import matplotlib.pyplot as plt

from itertools import chain
from skimage.morphology import label
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from sklearn.model_selection import train_test_split

# from tensorflow.keras.models import Model, load_model

import tensorflow as tf
import imshowpair
from collections import Counter
from focal_loss import BinaryFocalLoss
import pathlib
from tqdm import tqdm

# Library of training

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Dense, Dropout, experimental
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import gdown
import pkg_resources
from pathlib import Path

import subprocess
from multiprocessing import Process


BASE_DIR = Path(__file__).resolve().parents[1]

def _catch_error(f):
    @wraps(f)
    def wrap(*args,**kwargs):
        try:
            return f(*args,**kwargs)
        except Exception as e:
            raise web.HTTPBadRequest(reason=e)
    return wrap

def _fields_to_dict(fields_in):
    """
    Function to convert marshmallow fields to dict()
    """
    dict_out = {}
    for k, v in fields_in.items():
        param = {}
        param["default"] = v.missing
        param["type"] = type(v.missing)
        param["required"] = getattr(v, "required", False)

        v_help = v.metadata["description"]
        if "enum" in v.metadata.keys():
            v_help = f"{v_help}. Choices: {v.metadata['enum']}"
        param["help"] = v_help

        dict_out[k] = param

    return dict_out


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


def launch_cmd(logdir, port):
    subprocess.call(["tensorboard",
                     "--logdir", f"{logdir}",
                     "--port", f"{port}",
                     "--host", "0.0.0.0"])


def launch_tensorboard(logdir, port=6006):
    """
    Run Tensorboard on a separate Process on behalf of the user
    Parameters
    ==========
    * logdir: str, pathlib.Path
        Folder path to tensorboard logs.
    * port: int
        Port to use for the monitoring webserver.
    """
    subprocess.run(
        ["fuser", "-k", f"{port}/tcp"]  # kill any previous process in that port
    )
    p = Process(target=launch_cmd, args=(logdir, port), daemon=True)
    p.start()
    
# try:
#     mount_nextcloud('rshare:/data/dataset_files', paths.get_splits_dir())
#     mount_nextcloud('rshare:/data/images', paths.get_images_dir())
#     #mount_nextcloud('rshare:/models', paths.get_models_dir())
# except Exception as e:
#     print(e)

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

def dice_coefficient(y_true, y_pred):
    eps = 1e-6
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (keras.backend.sum(y_true_f * y_true_f) + keras.backend.sum(y_pred_f * y_pred_f) + eps) # eps pour éviter la division par 0

@_catch_error
def get_metadata():
    """
    DO NOT REMOVE - All modules should have a get_metadata() function
    with appropriate keys.
    """
    distros = list(pkg_resources.find_distributions(str(BASE_DIR), only=True))
    if len(distros) == 0:
        raise Exception("No package found.")
    pkg = distros[0]  # if several select first

    meta_fields = {
        "name": None,
        "version": None,
        "summary": None,
        "home-page": None,
        "author": None,
        "author-email": None,
        "license": None,
    }
    meta = {}
    for line in pkg.get_metadata_lines("PKG-INFO"):
        line_low = line.lower()  # to avoid inconsistency due to letter cases
        for k in meta_fields:
            if line_low.startswith(k + ":"):
                _, value = line.split(": ", 1)
                meta[k] = value

    return meta

def get_train_args():
    arg_dict = {
        # "image": fields.Field(
        #     required=False,
        #     type="file",
        #     missing="None",
        #     location="form",
        #     description="Compressed file .zip of images and masks",  # needed to be parsed by UI
        # ),
        "learning_rate": fields.Str(
            required = False,
            missing="0.0007",
            description="learning rate",
        ),
        "loss_function": fields.Str(
            required=False,  # force the user to define the value
            missing="weighted loss",  # default value to use
            enum=["weighted_loss", "focal_loss"],  # list of choices
            description="Loss function"  # help string
        ),
        "noyau": fields.Str(
            required = False,
            missing="3",
            description="Taille des noyau",
        ),
        "batch_size": fields.Str(
            required = False,
            missing="2",
            description="batch_size",
        ),
        "epoch": fields.Str(
            required = False,
            missing="50",
            description="Epoch",
        ),
        "gamma": fields.Str(
            required = False,
            missing="0.2",
            description="If focal loss selected, then choose your gamma value",
        ),
        "Link_images": fields.Str(
            required=False,
            missing="None",
            description="Link of image_user.zip located into google drive",  # needed to be parsed by UI
        ),
        "Link_model": fields.Str(
            required=False,
            missing="None",
            description="Link of best_models_.zip located into google drive",  # needed to be parsed by UI
        ),
    }
    return arg_dict

def train(**args):
    output={}
    output["hyperparameter"]=args
    backend.clear_session()
    
    print("Model downloading...")
    

    
    link_zip_file_images = yaml.safe_load(args["Link_images"])    
    # Images zip
    print("link_zip_file_images ",link_zip_file_images)
    
    try:
        
        mount_nextcloud('rshare:/data/dataset_files', paths.get_splits_dir())
        mount_nextcloud('rshare:/data/images', paths.get_images_dir())
        
    except Exception as e:
        print(e)
        if link_zip_file_images!="None":
            output_dir_images = tempfile.TemporaryDirectory()
            output_path_dir_images = output_dir_images.name
            
            id_file_images = link_zip_file_images.split('/')[-2]
            url_images = "https://drive.google.com/uc?export=download&id="+id_file_images
            output_zip_path = os.path.join(output_path_dir_images,'image_user.zip')
            print("Loading..")
            gdown.download(url_images, output_zip_path, quiet=False)
            # print(">>> output_dir_images",output_dir_images)
            zip_dir = tempfile.TemporaryDirectory()
            with ZipFile(output_zip_path,'r') as zipObject:
                listOfFileNames = zipObject.namelist()
                # print(listOfFileNames)
                for i in range(len(listOfFileNames)):
                    zipObject.extract(listOfFileNames[i],path=zip_dir.name)
            A1 = [os.path.join(zip_dir.name,ix) for ix in os.listdir(zip_dir.name)]            
            # print("A1 ",A1)
            verif = A1[0].split('\\')
            if verif[-1]=='images':
                path_image_data = A1[0]
                path_masks_data = A1[1]
            else:
                path_image_data = A1[1]
                path_masks_data = A1[0]        
        else:
            path_image_data = cfg.DATA_IMAGE
            path_masks_data = cfg.DATA_MASK

    # Model zip
    output_dir_model = tempfile.TemporaryDirectory()
    output_path_dir_model = output_dir_model.name
        
    link_zip_file_model = yaml.safe_load(args["Link_model"])
    id_file_model = link_zip_file_model.split('/')[-2]
    url_model = "https://drive.google.com/uc?export=download&id="+id_file_model
    
    output_zip_path = os.path.join(output_path_dir_model,'models_images.zip')
    print("Loading..")
    gdown.download(url_model, output_zip_path, quiet=False)
    # print(">>> output_dir_model",output_dir_model)
    with ZipFile(output_zip_path,'r') as zipObject:
        listOfFileNames = zipObject.namelist()
        for i in range(len(listOfFileNames)):
            zipObject.extract(listOfFileNames[i],path=output_path_dir_model)
    
    # print(os.listdir(output_path_dir_model))
    model_h5_path = os.path.join(output_path_dir_model,"best_model_W_BCE_model.h5")
    weight_h5_path = os.path.join(output_path_dir_model,"best_model.h5")
    opt_th_path = os.path.join(output_path_dir_model,"optimal_threshold.txt")
    if os.path.isfile(model_h5_path):
        print("best_model_W_BCE_model.h5 exist")
    else:
        print(" no best_model_W_BCE_model.h5 found")
    print("model downloaded !")
    
    output_zip_model_opt_thr = tempfile.TemporaryDirectory() # Where to save model h5 and opt txt
    output_zip_model_opt_thr_path_dir = output_zip_model_opt_thr.name

    def fabriquer_train(path):
        dico = {}
        A = os.listdir(path)
        for i in range(len(A)):
            img = imread(os.path.join(path,A[i]))
            dico[A[i]]=np.array(img)
        return dico

    def fabriquer_test(path):
        dico = {}
        A = os.listdir(path)
        for i in range(len(A)):
            img = imread(os.path.join(path,A[i]))
            dico[A[i]]=np.array(img)
        return dico

    print("train")
    image_ = fabriquer_train(path_image_data)
    print("test")
    masks_ = fabriquer_test(path_masks_data)
    print("Input done")

    images_set = list(image_.keys())
    masks_set = list(masks_.keys())
    
    images_set.sort()
    masks_set.sort()

    print("Train :",len(images_set),len(masks_set))

    train_list = []
    masks_list = []

    ct=0
    cp=0
    CP = []
    
    for x,y in tqdm(zip(images_set,masks_set),total = len(images_set), desc ="Processing"):
        # sample_image_train = imread(cfg.DATA_IMAGE+'\\'+x)[:,:,:3]
        # sample_maque_train = imread(cfg.DATA_MASK+'\\'+y)[:,:,:3]        
        sample_image_train = imread(path_image_data+'/'+x)[:,:,:3]        
        sample_maque_train = imread(path_masks_data+'/'+y)
        
        if sample_image_train.shape[0]==sample_maque_train.shape[0] and sample_image_train.shape[1]==sample_maque_train.shape[1]:
            train_list.append(x)
            masks_list.append(y)
            ct+=1
        else:
            cp+=1
            CP.append([x,y])
    print("Image et masque de même taille",ct)
    print("Image et masque de taille différente",cp)


    X_train, X_test, y_train, y_test = train_test_split(train_list, masks_list, test_size=0.2, random_state=42)

    # Set some parameters
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 3

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

    print('Getting and chipping train images ... ')

    #TRAIN
    print("Nombre image",len(X_train))
    print("Nombre masque",len(y_train))

    X_train_list = []
    y_train_list = []

    X_train_image = [path_image_data+'/'+files for files in X_train] #X_train <- images

    Y_train_masks = [path_masks_data+'/'+files for files in y_train] #Y_train <- masks

    X_test_images = [path_image_data+'/'+files for files in X_test] #X_test <- images

    y_test_masks = [path_masks_data+'/'+files for files in y_test] #Y_test <- masks

    for files_image,files_mask in tqdm(zip(X_train_image,Y_train_masks),total = len(X_train_image), desc ="Train processing"):

        img1 = imread(files_image)[:,:,:3]
        img2 = imread(files_mask)

        img1_list = get_mosaic(img1)
        img2_list = get_mosaic(img2)
        # on écarte les images avec un seul label
        for x,y in zip(img1_list,img2_list):
            sz1_x,sz2_x,sz3_x = x.shape
            y_shape = y.shape
            if len(y_shape)==3:
                sz1_y,sz2_y,sz3_y = y.shape
                # masque
                gray_file = rgb2gray(y)

                if len(Counter(gray_file.flatten()).keys())!=1:            
                    threshold = threshold_otsu(gray_file) #scikit-image 0.17.8 gray_file must have more than one value in matrix
                else:
                    threshold = list(Counter(gray_file.flatten()).keys())[0]
                binary_file = (gray_file > threshold)
                mask_ = np.expand_dims(binary_file, axis=-1)
                L = dict(Counter(list(mask_.flatten())))
                if len(list(L.keys()))==2 and (sz1_x,sz2_x)==(256,256) and (sz1_y,sz2_y)==(256,256):
                    X_train_list.append(x)
                    y_train_list.append(mask_)
            else:
                mask_ = np.expand_dims(y, axis=-1)
                L = dict(Counter(list(y.flatten())))
                if len(list(L.keys()))==2 and (sz1_x,sz2_x)==(256,256) and (y_shape[0],y_shape[1])==(256,256):
                    X_train_list.append(x)
                    y_train_list.append(mask_)
                


    print("Total image train for training step :")
    print("x_train :",len(X_train_list))
    print("y_train :",len(y_train_list))

    #TEST

    print("Total image in test sample",len(X_test))
    print("Total mask in test sample",len(y_test))

    X_test_list = []
    y_test_list = []

    for files_image,files_mask in tqdm(zip(X_test_images,y_test_masks),total = len(X_test_images), desc ="Test processing"):
        img1 = imread(files_image)[:,:,:3]
        img2 = imread(files_mask)
        img1_list = get_mosaic(img1)
        img2_list = get_mosaic(img2)

        #on écarte les images avec un seul label
        for x,y in zip(img1_list,img2_list):
            sz1_x,sz2_x,sz3_x = x.shape
            y_shape = y.shape
            if len(y_shape)==3:
                sz1_y,sz2_y,sz3_y = y.shape

                #masque
                gray_file = rgb2gray(y)
                
                if len(Counter(gray_file.flatten()).keys())!=1:            
                    threshold = threshold_otsu(gray_file) #scikit-image 0.17.8 gray_file must have more than one value in matrix
                else:
                    threshold = list(Counter(gray_file.flatten()).keys())[0]
                binary_file = (gray_file > threshold)
                mask_ = np.expand_dims(binary_file, axis=-1)

                L = dict(Counter(list(mask_.flatten())))
                if len(list(L.keys()))==2 and (sz1_x,sz2_x)==(256,256) and (sz1_y,sz2_y)==(256,256):
                    X_test_list.append(x)
                    y_test_list.append(mask_)
            else:
                L = dict(Counter(list(y.flatten())))
                mask_ = np.expand_dims(y, axis=-1)
                if len(list(L.keys()))==2 and (sz1_x,sz2_x)==(256,256) and (y_shape[0],y_shape[1])==(256,256):
                    X_test_list.append(x)
                    y_test_list.append(mask_)

    print("Total image test pour test step :")
    print("x_test :",len(X_test_list))
    print("y_test :",len(y_test_list))

    taille_p = 256
    X_train_ensemble = np.zeros((len(X_train_list), taille_p, taille_p, 3), dtype=np.uint8)
    y_train_ensemble = np.zeros((len(y_train_list), taille_p, taille_p, 1), dtype=np.bool)

    for n,m in zip(range(len(X_train_list)),range(len(y_train_list))):
        X_train_ensemble[n]=X_train_list[n]
        y_train_ensemble[m]=y_train_list[m]

    X_test_ensemble = np.zeros((len(X_test_list), taille_p, taille_p, 3), dtype=np.uint8)
    y_test_ensemble = np.zeros((len(y_test_list), taille_p, taille_p, 1), dtype=np.bool)

    for n,m in zip(range(len(X_test_list)),range(len(y_test_list))):
        X_test_ensemble[n]=X_test_list[n]
        y_test_ensemble[m]=y_test_list[m]

    x_train, x_val, y_train, y_val = train_test_split(X_train_ensemble, y_train_ensemble, test_size=0.2, random_state=42) 

    size_kernel_user = yaml.safe_load(args["noyau"])
    learning_rate_user = yaml.safe_load(args["learning_rate"])
    batch_size_user = yaml.safe_load(args["batch_size"])
    gamma_user = yaml.safe_load(args["gamma"])
    epoch_user = yaml.safe_load(args["epoch"])
    
    loss_function_user = yaml.safe_load(args["loss_function"])

    # MODEL LOADED..
    
    input_img = Input((256,256, 3), name='img')
    # model = get_unet(input_img, n_filters=n_filters_user, kernel_size=3) # nombre de filtre
    model = get_unet(input_img, n_filters=3, kernel_size=size_kernel_user) # nombre de filtre

    # model = tf.keras.models.load_model(os.path.join(paths.get_models_dir(),"best_model_W_BCE_model.h5"),custom_objects={'dice_coefficient': dice_coefficient})
    # model = tf.keras.models.load_model(model_h5_path,custom_objects={'dice_coefficient': dice_coefficient})
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_user)

    print('LOSS FUNCTION SELECTED :', loss_function_user)
    if loss_function_user=="weighted_loss":
        PIXEL = []
        for sample_y in y_train_ensemble: # image dans Y_train (masque de segmentation dans la phase d'entrainement)
            x = np.squeeze(sample_y)
        for i in range(255):
            for j in range(255):
                PIXEL.append(x[i][j])
        temp_dico = dict(Counter(PIXEL))
        V = temp_dico.values()
        s = sum(V)
        dico = {}
        for x in temp_dico:
            dico[x]=1-temp_dico[x]/s #<- vrai
        temp_list = list(dico.values())
    
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[dice_coefficient],loss_weights=temp_list) #weighted loss      
    else:
        model.compile(optimizer=opt, loss=[BinaryFocalLoss(gamma=gamma_user)], metrics=[dice_coefficient]) #weighted loss      

    model.load_weights(weight_h5_path)  
    # model.load_weights(model_h5_path)  

    model.summary() 

    
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto') #Stop training when a monitored metric has stopped improving.
    
    checkpoint_filepath = os.path.join(output_zip_model_opt_thr_path_dir,"output_best_model.h5")
    Model_check = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False, mode='auto') #Callback to save the Keras model or model weights at some frequency.
    print("training steps")
    print("total x_train :",len(x_train))
    print("total y_train :",len(y_train))
    print("total x_val :",len(x_val))
    print("total y_val :",len(y_val))
    results = model.fit(x_train,y_train,
                    validation_data=(x_val,y_val),
                    epochs=epoch_user, batch_size = batch_size_user,
                    callbacks=[early_stop,Model_check])


    # BEST RETRAIN MODEL
    print("best model analysis...")
    print("best model loading : output_best_model.h5")
    # model_New = tf.keras.models.load_model(os.path.join(paths.get_models_dir(),"output_best_model.h5"),custom_objects={'dice_coefficient': dice_coefficient})
    model_New = tf.keras.models.load_model(os.path.join(output_zip_model_opt_thr_path_dir,"output_best_model.h5"),custom_objects={'dice_coefficient': dice_coefficient})
    
    model_New.compile(optimizer=opt, loss="binary_crossentropy", metrics=[dice_coefficient],loss_weights=temp_list)
    eval_test=model_New.evaluate(X_test_ensemble,y_test_ensemble)
    Mask_valid_pred_int= model_New.predict(x_val, verbose=2)

    from sklearn.metrics import f1_score
    print("f1_score research...")
    prob_thresh = [i*10**-1 for i in range(1,10)]
    perf=[] # define an empty array to store the computed F1-score for each threshold
    perf_ALL=[]
    for r in prob_thresh:
        # print("step 1 loop")
        preds_bin = ((Mask_valid_pred_int> r) + 0 )
        preds_bin1=preds_bin[:,:,:,0]
        GTALL=y_val[:,:,:,0]
        for ii in range(len(GTALL)): # all validation images
            # print("step 2 loop")
            predmask=preds_bin1[ii,:,:]
            GT=GTALL[ii,:,:]
            l = GT.flatten()
            p= predmask.flatten()
            perf.append(f1_score(l, p)) # re invert the maps: cells: 1, back :0
        # print("step 1 end loop")
        perf_ALL.append(np.mean(perf))
        perf=[]

    max_f1 = max(perf_ALL)  # Find the maximum y value
    op_thr = prob_thresh[np.array(perf_ALL).argmax()]  # Find the x value corresponding to the maximum y value
    print (' Best threshold is:',op_thr, 'for F1-score=',max_f1)
    

    preds_test = model_New.predict(X_test_ensemble, verbose=1)
    # we apply a threshold on predicted mask (probability mask) to convert it to a binary mask.
    preds_test_opt = (preds_test >op_thr).astype(np.uint8)

    # save weight
    print("Weight model save : output_weight_best_model.h5")
    # save weight function of keras doesn't work
    # use instead save function
    # model_New.save(os.path.join(paths.get_models_dir(),"output_weight_best_model.h5"))
    model_New.save(os.path.join(output_zip_model_opt_thr_path_dir,"output_weight_best_model.h5"))
    
    
    # save op_thr in txt file
    # if os.path.exists(os.path.join(paths.get_models_dir(),"output_optimal_threshold.txt")):
    #     print("output_optimal_threshold.txt already exist... delete")
    #     os.remove(os.path.join(paths.get_models_dir(),"output_optimal_threshold.txt"))
    if os.path.exists(os.path.join(output_zip_model_opt_thr_path_dir,"output_optimal_threshold.txt")):
        print("output_optimal_threshold.txt already exist... delete")
        os.remove(os.path.join(output_zip_model_opt_thr_path_dir,"output_optimal_threshold.txt"))

    # f = open(os.path.join(paths.get_models_dir(),"output_optimal_threshold.txt"),"a")
    # f.write(str(op_thr))
    # f.close()
    # print("output_optimal_threshold.txt newly created")
    f = open(os.path.join(output_zip_model_opt_thr_path_dir,"output_optimal_threshold.txt"),"a")
    f.write(str(op_thr))
    f.close()
    print("output_optimal_threshold.txt newly created")

    PIXEL_TEST = []
    PIXEL_PRED = []
    for ix in range(len(X_test_ensemble)):
        a = y_test_ensemble[ix, :, :, 0]
        b = preds_test_opt[ix, :, :, 0]
        for i in range(256):
            for j in range(256):
                PIXEL_TEST.append(int(a[i][j]))
                PIXEL_PRED.append(int(b[i][j]))

    Y_t = keras.backend.constant(PIXEL_TEST)
    pred_t = keras.backend.constant(PIXEL_PRED)
    dice_retrain = keras.backend.get_value(dice_coefficient(Y_t,pred_t))
    m = tf.keras.metrics.MeanIoU(num_classes=2)
    m.update_state(Y_t,pred_t)
    jaccard = m.result().numpy()
    print("DICE RETRAIN :",dice_retrain)

    # TRUE MODEL
    # model_New = tf.keras.models.load_model(os.path.join(paths.get_models_dir(),"best_model_W_BCE_model.h5"),custom_objects={'dice_coefficient': dice_coefficient})
    model_New = tf.keras.models.load_model(model_h5_path,custom_objects={'dice_coefficient': dice_coefficient})
    model_New.compile(optimizer=opt, loss="binary_crossentropy", metrics=[dice_coefficient],loss_weights=temp_list)
    # model_New.compile(optimizer=opt, loss=[BinaryFocalLoss(gamma=gamma_user)], metrics=[dice_coefficient])

    eval_test=model_New.evaluate(X_test_ensemble,y_test_ensemble)

    Mask_valid_pred_int= model_New.predict(x_val, verbose=2)

    # get op_thr
    # f = open(os.path.join(paths.get_models_dir(),"optimal_threshold.txt"),"r")   
    f = open(opt_th_path,"r")
    numer_opt_thr = f.read()
    f.close()
    op_thr = float(numer_opt_thr)


    

    preds_test = model_New.predict(X_test_ensemble, verbose=1)
    # we apply a threshold on predicted mask (probability mask) to convert it to a binary mask.
    preds_test_opt = (preds_test >op_thr).astype(np.uint8)

    PIXEL_TEST = []
    PIXEL_PRED = []
    for ix in range(len(X_test_ensemble)):
        a = y_test_ensemble[ix, :, :, 0]
        b = preds_test_opt[ix, :, :, 0]
        for i in range(256):
            for j in range(256):
                PIXEL_TEST.append(int(a[i][j]))
                PIXEL_PRED.append(int(b[i][j]))

    Y_t = keras.backend.constant(PIXEL_TEST)
    pred_t = keras.backend.constant(PIXEL_PRED)
    dice_exist = keras.backend.get_value(dice_coefficient(Y_t,pred_t))
    m = tf.keras.metrics.MeanIoU(num_classes=2)
    m.update_state(Y_t,pred_t)
    jaccard = m.result().numpy()
    print("DICE EXISTED MODEL :",dice_exist)


    output["dice value (retrain model)"] = dice_retrain
    output["dice value (exist model)"] = dice_exist
    if dice_retrain <= dice_exist:
        output["retrain model"] = "worse"
    else:
        output["retrain model"] = "better"
    
    print("output zip files :",os.listdir(output_zip_model_opt_thr_path_dir))
    
    print(output)
    
    x = input('Do you want weight and model ? [Y/n]')
    if x=="" or x=='Y':
        gauth = GoogleAuth()           
        drive = GoogleDrive(gauth)  

        A = os.listdir(output_zip_model_opt_thr_path_dir) #contient seulement et uniquement des fichiers !!!
        # print(os.listdir(path_folder_to_image)) 
        PATH_mask = [os.path.join(output_zip_model_opt_thr_path_dir,ix) for ix in A]
        print(PATH_mask)

        id_output_folder = "1JMeVNJPrOtK13ZIqE8Q7R5PSqFpHRGFZ"
        for iy in PATH_mask:
            gfile = drive.CreateFile({'parents': [{'id': id_output_folder}]})
            gfile.SetContentFile(iy)
            gfile.Upload() # Upload the file.
    print("-x-")
    return output

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

def get_predict_args():
    """
    Input fields for the user (inference)
    """
    arg_dict = {
        "image": fields.Field(
            required=False,
            type="file",
            missing="None",
            location="form",
            description="Image",  # needed to be parsed by UI
        ),
        "Link": fields.Str(
            required=False,
            missing="None",
            description="Link of best_models_.zip located into google drive",  # needed to be parsed by UI
        ),
        "accept": fields.Str(
            description="Media type(s) that is/are acceptable for the response.",
            missing='application/zip',
            validate=validate.OneOf(['application/zip', 'image/png', 'application/json']),
        ),
    }
    return arg_dict


@_catch_error
def predict(**kwargs):
    """
    OUTPUT
    """
    print(kwargs)
    filepath = kwargs["image"].filename
    originalname = kwargs["image"].original_filename

    print(originalname)

    def dice_coefficient(y_true,y_pred):
        eps = 1e-6
        y_true_f = keras.backend.flatten(y_true)
        y_pred_f = keras.backend.flatten(y_pred)
        intersection =keras.backend.sum(y_true_f*y_pred_f)
        return (2. * intersection) / (keras.backend.sum(y_true_f * y_true_f) + keras.backend.sum(y_pred_f * y_pred_f) + eps)


    link_zip_file = kwargs["Link"] 
    id_file = link_zip_file.split('/')[-2]
    url = "https://drive.google.com/uc?export=download&id="+id_file

    if originalname[-3:] in ['JPG','jpg','png','PNG']:
        # Load model from gdrive
        output_dir_model = tempfile.TemporaryDirectory()
        output_path_dir = output_dir_model.name
        
        output_zip_path = os.path.join(output_path_dir,'models_images.zip')
        # print("Loading..")
        gdown.download(url, output_zip_path, quiet=False)
        # print(">>> output_dir_model",output_dir_model)
        with ZipFile(output_zip_path,'r') as zipObject:
            listOfFileNames = zipObject.namelist()
            for i in range(len(listOfFileNames)):
                zipObject.extract(listOfFileNames[i],path=output_path_dir)
        
        model_h5_path = os.path.join(output_path_dir,"best_model_W_BCE_model.h5")
        opt_th_path = os.path.join(output_path_dir,"optimal_threshold.txt")
        if os.path.isfile(model_h5_path):
            print("best_model_W_BCE_model.h5 exist")
        else:
            print(" no best_model_W_BCE_model.h5 found")
            
        f = open(opt_th_path,"r")
        numer_opt_thr = f.read()
        f.close()
        op_thr = float(numer_opt_thr)
        
        
        # Process data
        
        image_reshaped = imread(filepath)
        
        img1_list = get_mosaic_predict(image_reshaped)
        model_New = tf.keras.models.load_model(model_h5_path,custom_objects={'dice_coefficient': dice_coefficient})

        taille_p = 256
        X_ensemble = np.zeros((len(img1_list), taille_p, taille_p, 3), dtype=np.uint8)
        for n in range(len(img1_list)):
            sz1_x,sz2_x,sz3_x = img1_list[n].shape
            if (sz1_x,sz2_x)==(256,256):
                X_ensemble[n]=img1_list[n]

        # f = open(os.path.join(paths.get_models_dir(),"optimal_threshold.txt"),"r")
        # numer_opt_thr = f.read()
        # f.close()
        # op_thr = float(numer_opt_thr)

        preds_test = model_New.predict(X_ensemble, verbose=1)
        preds_test_opt = (preds_test > op_thr).astype(np.uint8)
        output_image = reconstruire(image_reshaped,preds_test_opt)
        
        output_dir = tempfile.TemporaryDirectory()
        imsave(fname=os.path.join(output_dir.name,originalname), arr=np.squeeze(output_image[:,:,0]))

        # return open(os.path.join(output_dir.name,originalname),'rb')
        shutil.make_archive(output_dir.name,format="zip",root_dir=output_dir.name,)
        zip_path = output_dir.name + ".zip"
        return open(zip_path,"rb")

    elif originalname[-3:] in ['zip','ZIP']:
        # Load model from gdrive
        output_dir_model = tempfile.TemporaryDirectory()
        output_path_dir = output_dir_model.name
        
        output_zip_path = os.path.join(output_path_dir,'models_images.zip')
        print("Loading..")
        gdown.download(url, output_zip_path, quiet=False)
        print(">>> output_dir_model",output_dir_model)
        with ZipFile(output_zip_path,'r') as zipObject:
            listOfFileNames = zipObject.namelist()
            for i in range(len(listOfFileNames)):
                zipObject.extract(listOfFileNames[i],path=output_path_dir)
        
        model_h5_path = os.path.join(output_path_dir,"best_model_W_BCE_model.h5")
        opt_th_path = os.path.join(output_path_dir,"optimal_threshold.txt")
        if os.path.isfile(model_h5_path):
            print("best_model_W_BCE_model.h5 exist")
        else:
            print(" no best_model_W_BCE_model.h5 found")
        
        # Load opt_th
        f = open(opt_th_path,"r")
        numer_opt_thr = f.read()
        f.close()
        op_thr = float(numer_opt_thr)
        
        
        zip_dir = tempfile.TemporaryDirectory()
        print(">>>>>>>>>>>>",zip_dir)
        with ZipFile(filepath,'r') as zipObject:
            listOfFileNames = zipObject.namelist()
            for i in range(len(listOfFileNames)):
                zipObject.extract(listOfFileNames[i],path=zip_dir.name)
        dico = {}
        for x in os.listdir(zip_dir.name):
            dico[x] = os.path.join(zip_dir.name,x)
        print(">>> 2")
        # Load model
        print(">>>> ==")
        # print(os.path.join(paths.get_models_dir(),'best_model_W_BCE_model.h5'))
        # model_New = tf.keras.models.load_model(os.path.join(paths.get_models_dir(),'best_model_W_BCE_model.h5'),custom_objects={'dice_coefficient': dice_coefficient})
        model_New = tf.keras.models.load_model(model_h5_path,custom_objects={'dice_coefficient': dice_coefficient})
        print(">>> 3")
        dico_prediction = {}
        output_dir = tempfile.TemporaryDirectory()

        # Load opt_th
        # f = open(os.path.join(paths.get_models_dir(),"optimal_threshold.txt"),"r")
        # numer_opt_thr = f.read()
        # print(">>>>>>>>>>>>",numer_opt_thr)
        # f.close()
        # op_thr = float(numer_opt_thr)

        for ids in list(dico.keys()):
            
            image_reshaped = imread(dico[ids])

            img1_list = get_mosaic_predict(image_reshaped) #dico = {image : [im_s1,im_s2,...]}

            taille_p = 256
            X_ensemble = np.zeros((len(img1_list), taille_p, taille_p, 3), dtype=np.uint8)

            for n in range(len(img1_list)):
                sz1_x,sz2_x,sz3_x = img1_list[n].shape
                if (sz1_x,sz2_x)==(256,256):
                    X_ensemble[n]=img1_list[n]
            
            preds_test = model_New.predict(X_ensemble, verbose=1)
            preds_test_opt = (preds_test > op_thr).astype(np.uint8)
            output_image = reconstruire(image_reshaped,preds_test_opt)
            
            # dico_prediction[ids] = np.squeeze(output_image[:,:,0])
            imsave(fname=os.path.join(output_dir.name,ids), arr=np.squeeze(output_image[:,:,0]))

        print(output_dir.name)
        shutil.make_archive(output_dir.name,format="zip",root_dir=output_dir.name,)
        zip_path = output_dir.name + ".zip"
        return open(zip_path,"rb")
