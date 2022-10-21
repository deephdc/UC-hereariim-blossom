import os.path
from datetime import datetime
# from blossom import config

# CONF = config.get_conf_dict()
homedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')

# base_dir = CONF['general']['base_directory']
base_dir = "."

def get_base_dir():
    return os.path.abspath(os.path.join(homedir, base_dir))

def get_models_dir():
    return os.path.join(get_base_dir(),os.path.join('blossom','models'))

def get_data_dir():
    return os.path.join(get_base_dir(), "dataset")

def get_timestamped_dir():
    return os.path.join(get_models_dir(), timestamp)

def get_logs_dir():
    return os.path.join(get_timestamped_dir(), "logs")