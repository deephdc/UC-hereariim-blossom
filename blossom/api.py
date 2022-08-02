from functools import wraps
import shutil
import tempfile
from marshmallow import missing
import yaml

from zipfile import ZipFile
from skimage.io import imread, imsave, imread_collection, concatenate_images

from tensorflow import keras
from focal_loss import BinaryFocalLoss
from tensorflow import keras

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
        "author":"xxx",
        "description":"xxx",
        "license":"MIT",
    }
    return metadata

def get_train_args():
    """
    Input fields for the user (inference)
    """
    arg_dict = {
        "filtre": fields.Int(
            required=False,
            missing=3,
        ),
        "learning_rate": fields.Float(
            required=False,
            missing=0.0007,
        ),
    }
    return arg_dict

def train(**args):
    output={}
    output["hyperparameter"]=args
    backend.clear_session()
    path_image_data = cfg.DATA_IMAGE
    path_masks_data = cfg.DATA_MASK

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
    
    image_name = list(image_.keys())
    masks_name = list(masks_.keys())

    
    X_train, X_test_, Y_train, Y_test_ = train_test_split(image_name, masks_name, test_size=0.2, random_state=42)

    x_train_, x_val_, y_train_, y_val_ = train_test_split(X_train, Y_train, test_size=0.2, random_state=42) #text_size=0.3 (à moi, 0.2)

    # Set some parameters
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 3

    def get_X_data(ids):
        ids.sort()
        X = np.zeros((len(ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        # for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        for n, id_ in enumerate(ids):
            # we'll be using skimage library for reading file
            print(len(image_))
            img = image_[id_][:,:,:IMG_CHANNELS]
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            X[n] = img
        return X

    def get_Y_data(ids):
        ids.sort()
        Y = np.zeros((len(ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        
        # for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        for n, id_ in enumerate(ids):
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
            print(len(masks_))
            masque_ = masks_[id_]
            gray_file = rgb2gray(masque_)
            threshold = threshold_otsu(gray_file)
            binary_file = (gray_file > threshold)
            masque_ = np.expand_dims(resize(binary_file, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True), axis=-1)
            Y[n] = masque_
        return Y

    print('Getting and resizing train images ... ')

    x_train = get_X_data(x_train_)
    y_train = get_Y_data(y_train_)

    x_val = get_X_data(x_val_)
    y_val = get_Y_data(y_val_)

    X_test = get_X_data(X_test_)
    Y_test = get_Y_data(Y_test_)

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

    n_filters_user = yaml.safe_load(args["filtre"])
    learning_rate_user = yaml.safe_load(args["learning_rate"])
    gamma_user = yaml.safe_load(args["gamma"])
    batch_size_user = yaml.safe_load(args["batch_size"])

    input_img = Input((256,256, 3), name='img')
    model = get_unet(input_img, n_filters=n_filters_user, kernel_size=3) #nombre de filtre

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_user)
    model.compile(optimizer=opt, loss=[BinaryFocalLoss(gamma=gamma_user)], metrics=[dice_coefficient]) #focal loss
    model.summary()


    model.load_weights(os.path.join(paths.get_models_dir(),"best_model_FL_BCE_0_5_model.h5"))

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, verbose=1, mode='auto') #Stop training when a monitored metric has stopped improving.

    # checkpoint_filepath = 'output_best_model.h5'
    checkpoint_filepath = os.path.join(paths.get_models_dir(),"output_best_model.h5")
    Model_check = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False, mode='auto') #Callback to save the Keras model or model weights at some frequency.

    results = model.fit(x_train,y_train, 
                    validation_data=(x_val,y_val),
                    epochs=50, batch_size = batch_size_user,
                    callbacks=[early_stop,Model_check])

    #RETRAIN
    model_New = tf.keras.models.load_model(os.path.join(paths.get_models_dir(),"output_best_model.h5"),custom_objects={'dice_coefficient': dice_coefficient})
    model_New.compile(optimizer=opt, loss=[BinaryFocalLoss(gamma=gamma_user)], metrics=[dice_coefficient]) 

    eval_test=model_New.evaluate(X_test,Y_test)

    Mask_valid_pred_int= model_New.predict(x_val, verbose=2)

    from sklearn.metrics import f1_score

    # compute F1-score for a set of thresholds from (0.1 to 0.9 with a step of 0.1)
    prob_thresh = [i*10**-1 for i in range(1,10)]
    perf=[] # define an empty array to store the computed F1-score for each threshold
    perf_ALL=[]
    # for r in tqdm(prob_thresh): # all th thrshold values
    for r in prob_thresh:
        preds_bin = ((Mask_valid_pred_int> r) + 0 )
        preds_bin1=preds_bin[:,:,:,0]
        GTALL=y_val[:,:,:,0]
        for ii in range(len(GTALL)): # all validation images
            predmask=preds_bin1[ii,:,:]
            GT=GTALL[ii,:,:]
            l = GT.flatten()
            p= predmask.flatten()
            perf.append(f1_score(l, p)) # re invert the maps: cells: 1, back :0
        perf_ALL.append(np.mean(perf))
        perf=[]
        
    max_f1 = max(perf_ALL)  # Find the maximum y value
    op_thr = prob_thresh[np.array(perf_ALL).argmax()]  # Find the x value corresponding to the maximum y value

    preds_test = model_New.predict(X_test, verbose=1)
    # we apply a threshold on predicted mask (probability mask) to convert it to a binary mask.
    preds_test_opt = (preds_test >op_thr).astype(np.uint8)

    PIXEL_TEST = []
    PIXEL_PRED = []
    for ix in range(len(X_test_)):
        a = Y_test[ix, :, :, 0]
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

    #TRUE MODEL
    model_New = tf.keras.models.load_model(os.path.join(paths.get_models_dir(),"best_model_FL_BCE_0_5_model.h5"),custom_objects={'dice_coefficient': dice_coefficient})
    model_New.compile(optimizer=opt, loss=[BinaryFocalLoss(gamma=gamma_user)], metrics=[dice_coefficient]) 

    eval_test=model_New.evaluate(X_test,Y_test)

    Mask_valid_pred_int= model_New.predict(x_val, verbose=2)

    

    # compute F1-score for a set of thresholds from (0.1 to 0.9 with a step of 0.1)
    prob_thresh = [i*10**-1 for i in range(1,10)]
    perf=[] # define an empty array to store the computed F1-score for each threshold
    perf_ALL=[]
    # for r in tqdm(prob_thresh): # all th thrshold values
    for r in prob_thresh:
        preds_bin = ((Mask_valid_pred_int> r) + 0 )
        preds_bin1=preds_bin[:,:,:,0]
        GTALL=y_val[:,:,:,0]
        for ii in range(len(GTALL)): # all validation images
            predmask=preds_bin1[ii,:,:]
            GT=GTALL[ii,:,:]
            l = GT.flatten()
            p= predmask.flatten()
            perf.append(f1_score(l, p)) # re invert the maps: cells: 1, back :0
        perf_ALL.append(np.mean(perf))
        perf=[]
        
    max_f1 = max(perf_ALL)  # Find the maximum y value
    op_thr = prob_thresh[np.array(perf_ALL).argmax()]  # Find the x value corresponding to the maximum y value

    preds_test = model_New.predict(X_test, verbose=1)
    # we apply a threshold on predicted mask (probability mask) to convert it to a binary mask.
    preds_test_opt = (preds_test >op_thr).astype(np.uint8)

    PIXEL_TEST = []
    PIXEL_PRED = []
    for ix in range(len(X_test_)):
        a = Y_test[ix, :, :, 0]
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
        preds_test_t = (prediction > 0.2)
        preds_test_t = resize(preds_test_t[0,:,:,0],(x,y),mode="constant",preserve_range=True)
        output_dir = tempfile.TemporaryDirectory()
        imsave(fname=os.path.join(output_dir.name,originalname), arr=np.squeeze(preds_test_t))

        # return open(os.path.join(output_dir.name,originalname),'rb')
        shutil.make_archive(output_dir.name,format="zip",root_dir=output_dir.name,)
        zip_path = output_dir.name + ".zip"
        return open(zip_path,"rb")
    
    elif originalname[-3:] in ['zip','ZIP']:
        zip_dir = tempfile.TemporaryDirectory()

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
            dico_size_ [ids] = size_

        model_new = tf.keras.models.load_model(os.path.join(paths.get_models_dir(),"best_model_FL_BCE_0_5_model.h5"),custom_objects={"dice_coefficient" : dice_coefficient})

        dico_prediction = {}
        output_dir = tempfile.TemporaryDirectory()

        for ids in dico.keys():
            prediction = model_new.predict(dico_image_reshaped[ids])
            x,y,z = dico_size_[ids]
            preds_test_t = (prediction > 0.2)
            preds_test_t = resize(preds_test_t[0,:,:,0],(x,y),mode="constant",preserve_range=True)
            dico_prediction[ids] = preds_test_t
            imsave(fname=os.path.join(output_dir.name,ids),arr=np.squeeze(preds_test_t))

        print(output_dir.name)
        shutil.make_archive(output_dir.name,format="zip",root_dir=output_dir.name,)
        zip_path = output_dir.name + ".zip"
        return open(zip_path,"rb")
