# -*- coding: utf-8 -*-
"""
   Module to define CONSTANTS used across the project
"""

import os
from webargs import fields
from marshmallow import Schema, INCLUDE
import yaml

# identify basedir for the package
BASE_DIR = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))

# default location for input and output data, e.g. directories 'data' and 'models',
# is either set relative to the application path or via environment setting
IN_OUT_BASE_DIR = BASE_DIR
if 'APP_INPUT_OUTPUT_BASE_DIR' in os.environ:
    env_in_out_base_dir = os.environ['APP_INPUT_OUTPUT_BASE_DIR']
    if os.path.isdir(env_in_out_base_dir):
        IN_OUT_BASE_DIR = env_in_out_base_dir
    else:
        msg = "[WARNING] \"APP_INPUT_OUTPUT_BASE_DIR=" + \
        "{}\" is not a valid directory! ".format(env_in_out_base_dir) + \
        "Using \"BASE_DIR={}\" instead.".format(BASE_DIR)
        print(msg)

DATA_DIR = os.path.join(IN_OUT_BASE_DIR,os.path.join('blossom','dataset'))
MODELS_DIR = os.path.join(IN_OUT_BASE_DIR, 'models')
DATA_IMAGE = os.path.join(DATA_DIR,os.path.join('train','images'))
DATA_MASK = os.path.join(DATA_DIR,os.path.join('train','masks'))

homedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
conf_path = os.path.join(homedir, 'etc', 'config.yaml')
with open(conf_path, 'r') as f:
    CONF = yaml.safe_load(f)

def get_conf_dict(conf=CONF):
    """
    Return configuration as dict
    """
    conf_d = {}
    for group, val in conf.items():
        conf_d[group] = {}
        for g_key, g_val in val.items():
            conf_d[group][g_key] = g_val['value']
    return conf_d


conf_dict = get_conf_dict()

def print_full_conf(conf=CONF):
    """
    Print all configuration parameters (including help, range, choices, ...)
    """
    for group, val in sorted(conf.items()):
        print('=' * 75)
        print('{}'.format(group))
        print('=' * 75)
        for g_key, g_val in sorted(val.items()):
            print('{}'.format(g_key))
            for gg_key, gg_val in g_val.items():
                print('{}{}'.format(' '*4, gg_key))
                body = '\n'.join(['\n'.join(textwrap.wrap(line, width=110, break_long_words=False,
                                                          replace_whitespace=False,
                                                          initial_indent=' '*8, subsequent_indent=' '*8))
                                  for line in str(gg_val).splitlines() if line.strip() != ''])
                print(body)
            print('\n')


def print_conf_table(conf=conf_dict):
    """
    Print configuration parameters in a table
    """
    print("{:<25}{:<30}{:<30}".format('group', 'key', 'value'))
    print('=' * 75)
    for group, val in sorted(conf.items()):
        for g_key, g_val in sorted(val.items()):
            print("{:<25}{:<30}{:<15} \n".format(group, g_key, str(g_val)))
        print('-' * 75 + '\n')

# Input parameters for predict() (deepaas>=1.0.0)
class PredictArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # support 'full_paths' parameter

    # full list of fields: https://marshmallow.readthedocs.io/en/stable/api_reference.html
    # to be able to upload a file for prediction
    files = fields.Field(
        required=False,
        missing=None,
        type="file",
        data_key="data",
        location="form",
        description="Select a file for the prediction"
    )

    # to be able to provide an URL for prediction
    urls = fields.Url(
        required=False,
        missing=None,
        description="Provide an URL of the data for the prediction"
    )
    
    # an input parameter for prediction
    arg1 = fields.Integer(
        required=False,
        missing=1,
        description="Input argument 1 for the prediction"
    )

# Input parameters for train() (deepaas>=1.0.0)
class TrainArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # support 'full_paths' parameter

    # available fields are e.g. fields.Integer(), fields.Str(), fields.Boolean()
    # full list of fields: https://marshmallow.readthedocs.io/en/stable/api_reference.html
    arg1 = fields.Integer(
        required=False,
        missing=1,
        description="Input argument 1 for training"
    )

#additional

