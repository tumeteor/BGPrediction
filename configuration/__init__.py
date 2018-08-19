import json
from os import path

_local_file = 'configuration/local_config.json'
_global_file = 'configuration/config.json'

def load_config(file_name):
    global data
    with open(file_name) as json_data_file:
        data = json.load(json_data_file)

def load_local_config():
    load_config(_local_file)

def load_global_config():
    load_config(_global_file)

# if local config is present, use that. Otherwise, use general config (using remote db server..)
if path.isfile('configuration/local_config.json'):
    load_local_config()
else:
    load_global_config()
