import os.path
# from blossom import config

# CONF = config.get_conf_dict()
homedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# base_dir = CONF['general']['base_directory']
base_dir = "."

def get_base_dir():
    return os.path.abspath(os.path.join(homedir, base_dir))

def get_models_dir():
    return os.path.join(get_base_dir(),os.path.join('blossom','models'))

def get_data_dir():
    return os.path.join(get_base_dir(),'blossom',"dataset")

def get_rconf_file():
    return os.path.join(get_base_dir(),"rclone.conf")

def get_logs_dir():
    return os.path.join(get_models_dir(),"logs")

print(">",os.path.abspath(os.path.join(homedir, base_dir)))
print(">",os.path.join(get_base_dir(),os.path.join('blossom','models')))
print(">",os.path.join(get_base_dir(),os.path.join('blossom','dataset')))
print(">",os.path.join(get_base_dir(),"rclone.conf"))