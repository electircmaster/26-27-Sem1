import importlib.util
import json
import uuid
from pathlib import Path
import torch
from torch import nn
from .spec import SCHEMA, TF_CONFIG, MAX_CNN_PARAMETERS, MAX_WEIGHTS_BYTES
from .models import TinyTransformer


def read_config(directory):
    path=Path(directory)/'config.json'
    if path.stat().st_size>16_384: raise ValueError('config.json is too large.')
    config=json.loads(path.read_text())
    if set(config)!={'schema','cnn','transformer'} or config['schema']!=SCHEMA:
        raise ValueError('Use the supplied config schema: schema, cnn, transformer.')
    if not isinstance(config['cnn'],dict) or not isinstance(config['transformer'],dict):
        raise ValueError('Model configurations must be JSON objects.')
    return config


def import_student(directory,task,source=None):
    path=Path(directory)/f'{task}.py'
    if path.stat().st_size>200_000: raise ValueError('Model source exceeds 200 KB.')
    spec=importlib.util.spec_from_file_location(f'a3_{task}_{uuid.uuid4().hex}',path)
    module=importlib.util.module_from_spec(spec)
    source=path.read_bytes() if source is None else source
    if len(source)>200_000:raise ValueError('Model source exceeds 200 KB.')
    exec(compile(source,str(path),'exec'),module.__dict__)
    return module


def load_transformer_functions(directory,source=None):
    config=read_config(directory)
    if config['transformer']!=TF_CONFIG:raise ValueError('Transformer config must remain unchanged.')
    module=import_student(directory,'transformer',source=source)
    for name in ['scaled_dot_product_attention','make_causal_mask','prepare_inputs_and_targets']:
        if not callable(getattr(module,name,None)):raise TypeError(f'Missing callable: {name}')
    return module


def transformer_from_functions(module):
    # Always construct the course architecture. No student build_model() is called.
    return TinyTransformer(dict(TF_CONFIG),module.scaled_dot_product_attention,module.make_causal_mask)


def construct_model(directory,task,config=None):
    if task=='transformer':
        module=load_transformer_functions(directory)
        return module,transformer_from_functions(module)
    config=read_config(directory) if config is None else config
    module=import_student(directory,task);model=module.build_model(dict(config[task]))
    if not isinstance(model,nn.Module):raise TypeError('build_model must return nn.Module.')
    if any(p.device.type!='cpu' for p in model.parameters()):raise ValueError('Construct the CNN on CPU; the caller selects the device.')
    count=sum(p.numel() for p in model.parameters())
    if not 0<count<=MAX_CNN_PARAMETERS:raise ValueError(f'CNN has {count} parameters; limit is {MAX_CNN_PARAMETERS}.')
    if not any(isinstance(m,nn.Conv2d) for m in model.modules()):raise ValueError('CNN must use Conv2d.')
    if sum(t.numel() for t in model.state_dict().values())>2_000_000:raise ValueError('Model state is too large.')
    return module,model


def load_model(directory,task,device='cpu'):
    if task!='cnn':raise ValueError('Only CNN weights are submitted. Format models are trained by the course runtime.')
    module,model=construct_model(directory,task)
    name='cnn_weights.pt'
    path=Path(directory)/name
    if path.stat().st_size>MAX_WEIGHTS_BYTES: raise ValueError('Weights file exceeds 8 MB.')
    state=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(state,dict) or not all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items()):
        raise ValueError('Save model.state_dict(), not the entire model or a training bundle.')
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Weights contain nonfinite values.')
    model.load_state_dict(state,strict=True)
    return module,model.to(device).eval()
