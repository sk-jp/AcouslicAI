from addict import Dict
import yaml


def check_value(data, depth=0):
    for key, value in data.items():
        if value == "none" or value == "None":
            data[key] = None
        if isinstance(value, dict):
            check_value(value, depth + 1)
    return data
            

def read_yaml(fpath='./model.yaml'):
    with open(fpath, mode='r') as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        yml = check_value(yml)
            
        return Dict(yml)


if __name__ == '__main__':
    d = read_yaml('./unet_multitask.yaml')
    
    for key in d['Model'].keys():
        print(d['Model'][key])

