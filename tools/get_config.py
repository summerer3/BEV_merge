import json

def get_config(config_path):
    f=open(config_path,'r')
    js=json.load(f)
    cam_list = js['cameras']
    chess = js['chess']
    
    return cam_list,chess