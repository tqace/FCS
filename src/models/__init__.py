from .stage1 import InphyGPT

def build_model(params):
    if params['name'] == 'InphyGPT':
        return InphyGPT(
                n_preds = params['n_preds'],
                T = params['T'],
                K = params['K'],
                beta = params['beta'], 
                a_dim = params['a_dim'],
                resolution = params['resolution'],
                use_feature_extractor = params['use_feature_extractor'],
                t_pe = params['t_pe'],
                history_len = params['history_len'],
                d_model = params['d_model'],
                num_layers = params['num_layers'],
                num_heads = params['num_heads'],
                ffn_dim = params['ffn_dim'],
                )
    else:
        raise NotImplementedError("This model is not implemented yes.")

