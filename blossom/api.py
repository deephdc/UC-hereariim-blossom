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
        # "accept": fields.Str(
        #     description="Media type(s) that is/are acceptable for the response.",
        #     missing='application/zip',
        #     validate=validate.OneOf(['application/zip', 'image/png', 'application/json']),
        # ),
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
        print("2")
        print("--")
        model_new = tf.keras.models.load_model("./blossom/blossom/models/best_model_FL_BCE_0_5_model.h5",custom_objects={"dice_coefficient" : dice_coefficient})
        print("3")
        prediction = model_new.predict(image_reshaped)
        print("4")
        preds_test_t = (prediction > 0.2)
        print("5")
        preds_test_t = resize(preds_test_t[0,:,:,0],(x,y),mode="constant",preserve_range=True)
        print("6")
        output_dir = tempfile.TemporaryDirectory()
        imsave(fname=os.path.join(output_dir.name,originalname), arr=np.squeeze(preds_test_t))
        print("SAVE")
        return open(os.path.join(output_dir.name,originalname),'rb')
    
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

        model_new = keras.models.load_model("./blossom/blossom/models/best_model_FL_BCE_0_5_model.h5",custom_objects={"dice_coefficient" : dice_coefficient})

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
        zip_path = zip_dir.name + ".zip"
        return open(zip_path,"rb")