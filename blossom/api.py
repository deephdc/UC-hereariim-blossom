from functools import wraps
import shutil
import tempfile
from marshmallow import missing
import yaml
from webargs import fields

#127.0.0.1

from zipfile import ZipFile
from skimage.io import imread, imsave, imread_collection, concatenate_images

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
import os
import sys
import random
import warnings

import numpy as np
import imageio
import matplotlib.pyplot as plt

from itertools import chain
from skimage.transform import resize
from skimage.morphology import label
from skimage.color import rgb2gray


# from tensorflow.keras.models import Model, load_model

import tensorflow as tf
import imshowpair
from collections import Counter
from focal_loss import BinaryFocalLoss
import pathlib

# Library of training

from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Dense, Dropout, experimental
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from skimage.filters import threshold_otsu


import tensorflow as tf
import os

import tensorflow as tf

#This code helps if you are using GPU, else you can comment it.

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.compat.v1.Session(config=config)

def _catch_error(f):
    @wraps(f)
    def wrap(*args,**kwargs):
        try:
            return f(*args,**kwargs)
        except Exception as e:
            raise web.HTTPBadRequest(reason=e)
    return wrap

def get_metadata():
    metadata = {
        "name": fields.Str(required=True,
                       description='Blossom'),
        "version": fields.Str(required=False,
                          description='Model version'),
        "summary": fields.Str(required=False,
                         description='This module gives you a model to segment blossoming apple tree'),
        #"home-page": None,
        "author": fields.Str(required=False,
                         description='Herearii Metuarea'),
        "author-email": "herearii.metuarea@gmail.com",
        #"license": "MIT",
        "license": fields.Str(required=False,
                          description='MIT'),
    }
    return metadata

def get_train_args():
    arg_dict = {
        "learning_rate": fields.Str(
            required = False,
            missing="0.0007",
            description="learning rate",
        ),
        "filtre": fields.Str(
            required = False,
            missing="3",
            description="filtre",
        ),
        "gamma": fields.Str(
            required = False,
            missing="0.2",
            description="gamma",
        ),
        "batch_size": fields.Str(
            required = False,
            missing="2",
            description="batch_size",
        ),
    }
    return arg_dict

def train(**args):
    output={}
    output["hyperparameter"]=args
    backend.clear_session()
    path_image_data = cfg.DATA_IMAGE
    path_masks_data = cfg.DATA_MASK
    print("path image data",path_image_data)
    print("path mask data",path_masks_data)

    def fabriquer_train(path):
        dico = {}
        A = os.listdir(path)
        # for i in tqdm(range(len(A)),'train'):
        for i in range(len(A)):
            img = imread(os.path.join(path,A[i]))
            dico[A[i]]=np.array(img)
        return dico

    def fabriquer_test(path):
        dico = {}
        A = os.listdir(path)
        # for i in tqdm(range(len(A)),'test'):
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

    print("Train :",len(images_set),len(masks_set))

    train_list = []
    masks_list = []

    ct=0
    cp=0
    CP = []
    for x,y in zip(images_set,masks_set):
        sample_image_train = imread(cfg.DATA_IMAGE+'\\'+x)[:,:,:3]
        sample_maque_train = imread(cfg.DATA_MASK+'\\'+y)[:,:,:3]
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
        if train:
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

    print('Getting and resizing train images ... ')

    #TRAIN
    print("Nombre image",len(X_train))
    print("Nombre masque",len(y_train))

    X_train_list = []
    y_train_list = []

    X_train_image = [cfg.DATA_IMAGE+'\\'+files for files in X_train] #X_train <- images

    Y_train_masks = [cfg.DATA_MASK+'\\'+files for files in y_train] #Y_train <- masks

    X_test_images = [cfg.DATA_IMAGE+'\\'+files for files in X_test] #X_test <- images

    y_test_masks = [cfg.DATA_MASK+'\\'+files for files in y_test] #Y_test <- masks

    for files_image,files_mask in zip(X_train_image,Y_train_masks):
        print(files_image,files_mask)
        print("step 1")
        print(files_image)
        print(files_mask)
        img1 = imread(files_image)[:,:,:3]
        img2 = imread(files_mask)[:,:,:3]
        print("step 2")
        img1_list = get_mosaic(img1)
        img2_list = get_mosaic(img2)
        print("done")

        #on écarte les images avec un seul label
        for x,y in zip(img1_list,img2_list):
            sz1_x,sz2_x,sz3_x = x.shape
            sz1_y,sz2_y,sz3_y = y.shape

            #masque
            gray_file = rgb2gray(y)
            threshold = threshold_otsu(gray_file)
            binary_file = (gray_file > threshold)
            mask_ = np.expand_dims(binary_file, axis=-1)

            L = dict(Counter(list(mask_.flatten())))
            if len(list(L.keys()))==2 and (sz1_x,sz2_x)==(256,256) and (sz1_y,sz2_y)==(256,256):
                X_train_list.append(x)
                y_train_list.append(mask_)

    print("Total image train pour training step :")
    print("x_train :",len(X_train_list))
    print("y_train :",len(y_train_list))

    #TEST

    print("Nombre image",len(X_test))
    print("Nombre masque",len(y_test))

    X_test_list = []
    y_test_list = []

    for files_image,files_mask in zip(X_test_images,y_test_masks):
        img1 = imread(files_image)[:,:,:3]
        img2 = imread(files_mask)[:,:,:3]
        img1_list = get_mosaic(img1)
        img2_list = get_mosaic(img2)

        #on écarte les images avec un seul label
        for x,y in zip(img1_list,img2_list):
            sz1_x,sz2_x,sz3_x = x.shape
            sz1_y,sz2_y,sz3_y = y.shape

            #masque
            gray_file = rgb2gray(y)
            threshold = threshold_otsu(gray_file)
            binary_file = (gray_file > threshold)
            mask_ = np.expand_dims(binary_file, axis=-1)

            L = dict(Counter(list(mask_.flatten())))
            if len(list(L.keys()))==2 and (sz1_x,sz2_x)==(256,256) and (sz1_y,sz2_y)==(256,256):
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

        c4 = conv2d_block(p14, n_filters=n_filters*32, kernel_size=3)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

        c5 = conv2d_block(p4, n_filters=n_filters*64, kernel_size=3) # last layer on encoding path

        # expansive path # decoder
        u6 = Conv2DTranspose(n_filters*32, (3, 3), strides=(2, 2), padding='same') (c5) #upsampling included
        u6 = concatenate([u6, c4])
        c6 = conv2d_block(u6, n_filters=n_filters*32, kernel_size=3)

        u15 = Conv2DTranspose(n_filters*16, (3, 3), strides=(2, 2), padding='same') (c6)
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
        model = tf.keras.models.Model(inputs=[input_img], outputs=[outputs])
        return model

    def dice_coefficient(y_true, y_pred):
        eps = 1e-6
        y_true_f = keras.backend.flatten(y_true)
        y_pred_f = keras.backend.flatten(y_pred)
        intersection = keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection) / (keras.backend.sum(y_true_f * y_true_f) + keras.backend.sum(y_pred_f * y_pred_f) + eps) #eps pour éviter la division par 0

    x_train, x_val, y_train, y_val = train_test_split(X_train_ensemble, y_train_ensemble, test_size=0.2, random_state=42) 

    n_filters_user = yaml.safe_load(args["filtre"])
    learning_rate_user = yaml.safe_load(args["learning_rate"])
    gamma_user = yaml.safe_load(args["gamma"])
    batch_size_user = yaml.safe_load(args["batch_size"])

    input_img = Input((256,256, 3), name='img')
    model = get_unet(input_img, n_filters=n_filters_user, kernel_size=3) #nombre de filtre

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_user)
    model.compile(optimizer=opt, loss=[BinaryFocalLoss(gamma=gamma_user)], metrics=[dice_coefficient]) #focal loss
    model.summary()


    model.load_weights(os.path.join(paths.get_models_dir(),"weight_best_model_FL_BCE_0_5_model.h5"))

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, verbose=1, mode='auto') #Stop training when a monitored metric has stopped improving.

    # checkpoint_filepath = 'output_best_model.h5'
    checkpoint_filepath = os.path.join(paths.get_models_dir(),"output_best_model.h5")
    Model_check = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False, mode='auto') #Callback to save the Keras model or model weights at some frequency.
    print("training steps")
    results = model.fit(x_train,y_train,
                    validation_data=(x_val,y_val),
                    epochs=50, batch_size = batch_size_user,
                    callbacks=[early_stop,Model_check])


    #RETRAIN
    print("best model analysis...")
    print("best model loading : output_best_model.h5")
    model_New = tf.keras.models.load_model(os.path.join(paths.get_models_dir(),"output_best_model.h5"),custom_objects={'dice_coefficient': dice_coefficient})
    model_New.compile(optimizer=opt, loss=[BinaryFocalLoss(gamma=gamma_user)], metrics=[dice_coefficient])

    eval_test=model_New.evaluate(X_test_ensemble,y_test_ensemble)

    Mask_valid_pred_int= model_New.predict(x_val, verbose=2)

    from sklearn.metrics import f1_score
    print("f1_score research...")
    # compute F1-score for a set of thresholds from (0.1 to 0.9 with a step of 0.1)
    prob_thresh = [i*10**-1 for i in range(1,10)]
    perf=[] # define an empty array to store the computed F1-score for each threshold
    perf_ALL=[]
    # for r in tqdm(prob_thresh): # all th thrshold values
    for r in prob_thresh:
        print("step 1 loop")
        preds_bin = ((Mask_valid_pred_int> r) + 0 )
        preds_bin1=preds_bin[:,:,:,0]
        GTALL=y_val[:,:,:,0]
        for ii in range(len(GTALL)): # all validation images
            print("step 2 loop")
            predmask=preds_bin1[ii,:,:]
            GT=GTALL[ii,:,:]
            l = GT.flatten()
            p= predmask.flatten()
            perf.append(f1_score(l, p)) # re invert the maps: cells: 1, back :0
        print("step 1 end loop")
        perf_ALL.append(np.mean(perf))
        perf=[]

    max_f1 = max(perf_ALL)  # Find the maximum y value
    op_thr = prob_thresh[np.array(perf_ALL).argmax()]  # Find the x value corresponding to the maximum y value
    print (' Best threshold is:',op_thr, 'for F1-score=',max_f1)
    
    #last blocked <=====
    preds_test = model_New.predict(X_test_ensemble, verbose=1)
    # we apply a threshold on predicted mask (probability mask) to convert it to a binary mask.
    preds_test_opt = (preds_test >op_thr).astype(np.uint8)

    #save weight
    print("Weight model save : weight_output_best_model.h5")
    model_New.save("weight_output_best_model.h5")
    #save op_thr in txt file
    if os.path.exists(os.path.join(paths.get_models_dir(),"output_optimal_threshold.txt")):
        print("output_optimal_threshold.txt already exist... delete")
        os.remove(os.path.join(paths.get_models_dir(),"output_optimal_threshold.txt"))

    f = open(os.path.join(paths.get_models_dir(),"output_optimal_threshold.txt"),"a")
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

    #TRUE MODEL
    model_New = tf.keras.models.load_model(os.path.join(paths.get_models_dir(),"best_model_FL_BCE_0_5_model.h5"),custom_objects={'dice_coefficient': dice_coefficient})
    model_New.compile(optimizer=opt, loss=[BinaryFocalLoss(gamma=gamma_user)], metrics=[dice_coefficient])

    eval_test=model_New.evaluate(X_test_ensemble,y_test_ensemble)

    Mask_valid_pred_int= model_New.predict(x_val, verbose=2)

    # compute F1-score for a set of thresholds from (0.1 to 0.9 with a step of 0.1)
    # prob_thresh = [i*10**-1 for i in range(1,10)]
    # perf=[] # define an empty array to store the computed F1-score for each threshold
    # perf_ALL=[]
    # for r in tqdm(prob_thresh): # all th thrshold values
    # for r in prob_thresh:
    #     preds_bin = ((Mask_valid_pred_int> r) + 0 )
    #     preds_bin1=preds_bin[:,:,:,0]
    #     GTALL=y_val[:,:,:,0]
    #     for ii in range(len(GTALL)): # all validation images
    #         predmask=preds_bin1[ii,:,:]
    #         GT=GTALL[ii,:,:]
    #         l = GT.flatten()
    #         p= predmask.flatten()
    #         perf.append(f1_score(l, p)) # re invert the maps: cells: 1, back :0
    #     perf_ALL.append(np.mean(perf))
    #     perf=[]

    # max_f1 = max(perf_ALL)  # Find the maximum y value
    # op_thr = prob_thresh[np.array(perf_ALL).argmax()]  # Find the x value corresponding to the maximum y value


    # get op_thr
    f = open(os.path.join(paths.get_models_dir(),"optimal_threshold.txt"),"r")
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
    if dice_retrain < dice_exist:
        output["retrain model"] = "worse"
    else:
        output["retrain model"] = "better"
    print(output)
    return output


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

    filepath = kwargs["image"].filename
    originalname = kwargs["image"].original_filename

    print(originalname)

    def redimension(image):
        X = np.zeros((1,256,256,3),dtype=np.uint8)
        img = imread(image)
        size_ = img.shape
        X[0] = resize(img, (256,256), mode="constant", preserve_range=True)
        return X,size_

    def dice_coefficient(y_true,y_pred):
        eps = 1e-6
        y_true_f = keras.backend.flatten(y_true)
        y_pred_f = keras.backend.flatten(y_pred)
        intersection =keras.backend.sum(y_true_f*y_pred_f)
        return (2. * intersection) / (keras.backend.sum(y_true_f * y_true_f) + keras.backend.sum(y_pred_f * y_pred_f) + eps)

    if originalname[-3:] in ['JPG','jpg','png','PNG']:

        image_reshaped, size_ = redimension(filepath)
        x,y,z = size_
        model_new = tf.keras.models.load_model(os.path.join(paths.get_models_dir(),"best_model_FL_BCE_0_5_model.h5"),custom_objects={"dice_coefficient" : dice_coefficient})
        prediction = model_new.predict(image_reshaped)

        f = open(os.path.join(paths.get_models_dir(),"optimal_threshold.txt"),"r")
        numer_opt_thr = f.read()
        f.close()

        op_thr = float(numer_opt_thr)

        preds_test_t = (prediction > op_thr) #op_thr = 0.2
        preds_test_t = resize(preds_test_t[0,:,:,0],(x,y),mode="constant",preserve_range=True)
        output_dir = tempfile.TemporaryDirectory()
        imsave(fname=os.path.join(output_dir.name,originalname), arr=np.squeeze(preds_test_t))

        # return open(os.path.join(output_dir.name,originalname),'rb')
        shutil.make_archive(output_dir.name,format="zip",root_dir=output_dir.name,)
        zip_path = output_dir.name + ".zip"
        return open(zip_path,"rb")

    elif originalname[-3:] in ['zip','ZIP']:
        zip_dir = tempfile.TemporaryDirectory()
        print(">>>>>>>>>>>>",zip_dir)
        with ZipFile(filepath,'r') as zipObject:
            listOfFileNames = zipObject.namelist()
            for i in range(len(listOfFileNames)):
                zipObject.extract(listOfFileNames[i],path=zip_dir.name)

        dico = {}
        for x in os.listdir(zip_dir.name):
            dico[x] = os.path.join(zip_dir.name,x)

        dico_image_reshaped = {}
        dico_size_ = {}
        for ids in list(dico.keys()):
            image_reshaped, size_ = redimension(dico[ids])
            dico_image_reshaped[ids] = image_reshaped
            dico_size_[ids] = size_

        model_new = tf.keras.models.load_model(os.path.join(paths.get_models_dir(),"best_model_FL_BCE_0_5_model.h5"),custom_objects={"dice_coefficient" : dice_coefficient})

        dico_prediction = {}
        output_dir = tempfile.TemporaryDirectory()

        f = open(os.path.join(paths.get_models_dir(),"optimal_threshold.txt"),"r")
        numer_opt_thr = f.read()
        print(">>>>>>>>>>>>",numer_opt_thr)
        f.close()

        op_thr = float(numer_opt_thr)


        for ids in dico.keys():
            prediction = model_new.predict(dico_image_reshaped[ids])
            x,y,z = dico_size_[ids]
            preds_test_t = (prediction > op_thr) #op_thr = 0.2
            preds_test_t = resize(preds_test_t[0,:,:,0],(x,y),mode="constant",preserve_range=True)
            dico_prediction[ids] = preds_test_t
            imsave(fname=os.path.join(output_dir.name,ids),arr=np.squeeze(preds_test_t))

        print(output_dir.name)
        shutil.make_archive(output_dir.name,format="zip",root_dir=output_dir.name,)
        zip_path = output_dir.name + ".zip"
        return open(zip_path,"rb")
