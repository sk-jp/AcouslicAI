from collections import OrderedDict

def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
#        print('name:', name)
        if name.startswith('model.'):
            # remove 'model.'
            name = name[6:]
        elif name.startswith('target_model.'):
            # remove 'target_model.'
            name = name[13:]
        else:
            # skip if it's not started with 'model'
            continue
        new_state_dict[name] = v
    return new_state_dict
