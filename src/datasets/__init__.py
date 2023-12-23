from .data import build_comphyVid_dataset 

def build_dataset(config):
    args = {
        'data_root':config['data_root']
        }
    if config['name'] == 'comphyVid':
        return build_comphyVid_dataset(args)
    else:
        raise NotImplementedError("This dataset is not implemented yet.")



