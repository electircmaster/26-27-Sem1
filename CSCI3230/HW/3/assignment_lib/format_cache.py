"""Cache experiments by student source, format, fixed policy and runtime versions."""
import hashlib,json,os,platform,tempfile,pickle
from filelock import FileLock
from pathlib import Path
import numpy as np
import torch
from .spec import TF_CONFIG,TF_TRAIN_STEPS,TF_SEED
from .format_spec import canonical_format,validate_format
from .format_runtime import train_fixed,evaluate_fixed
from .loading import load_transformer_functions,transformer_from_functions
from .transformer_checks import preflight,StudentImplementationError


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def experiment_identity(spec,student_source):
    root=Path(__file__).parent
    sources={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
             ['format_runtime.py','format_spec.py','spec.py','models.py','data.py','format_cache.py','loading.py','transformer_checks.py']}
    return dict(protocol='student_transformer_v1',student_source_sha256=hashlib.sha256(student_source).hexdigest(),format=validate_format(spec),model=TF_CONFIG,steps=TF_TRAIN_STEPS,seed=TF_SEED,
                torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),device='cpu',threads=2,sources=sources)

def atomic_json(path,value):
    fd,name=tempfile.mkstemp(prefix='.pending_',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2)
        os.replace(name,path)
    finally:
        if Path(name).exists():Path(name).unlink()

def run_experiment(spec,cache_root,submission,samples=512,seed=2026,split='validation'):
    spec=validate_format(spec)
    if split not in ['validation','test'] or not 1<=samples<=10000:raise ValueError('Invalid evaluation profile.')
    torch.set_num_threads(2)
    try:
        source=(Path(submission)/'transformer.py').read_bytes()
        module=load_transformer_functions(submission,source=source)
        preflight(module,spec)
    except StudentImplementationError:raise
    except Exception as error:raise StudentImplementationError(f'Loading transformer.py: {error}') from error
    identity=experiment_identity(spec,source);key=digest(identity)
    root=Path(cache_root);root.mkdir(parents=True,exist_ok=True)
    directory=root/key;directory.mkdir(exist_ok=True)
    weights=directory/'weights.pt';metadata=directory/'training.json'
    eval_profile=dict(samples=samples,seed=seed,split=split)
    result_path=directory/f'eval_{digest(eval_profile)}.json'
    # Only identical implementation bytes and format share a training entry.
    with FileLock(str(root/f'{key}.lock')):
        trained_cache=False;model=None;info=None
        try:
            saved=json.loads(metadata.read_text())
            if saved['identity']!=identity or saved['weights_sha256']!=hashlib.sha256(weights.read_bytes()).hexdigest():raise ValueError('Stale cache.')
            model=transformer_from_functions(module);state=torch.load(weights,map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True)
            if any(v.is_floating_point() and not torch.isfinite(v).all() for v in state.values()):raise ValueError('Invalid cached weights.')
            info=saved['training'];trained_cache=True
        except (OSError,ValueError,KeyError,TypeError,RuntimeError,EOFError,pickle.UnpicklingError):
            model,info=train_fixed(spec,module)
            fd,name=tempfile.mkstemp(prefix='pending_weights_',suffix='.pt',dir=directory);os.close(fd)
            try:
                torch.save(model.state_dict(),name);os.replace(name,weights)
            finally:
                if Path(name).exists():Path(name).unlink()
            atomic_json(metadata,dict(identity=identity,training=info,weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest()))
        metrics_cache=False
        try:
            saved=json.loads(result_path.read_text())
            if not trained_cache or saved['profile']!=eval_profile or saved['sha256']!=digest(saved['metrics']):raise ValueError('Stale metric cache.')
            metrics=saved['metrics'];metrics_cache=True
        except (OSError,ValueError,KeyError,TypeError):
            metrics=evaluate_fixed(model,spec,**eval_profile)
            atomic_json(result_path,dict(profile=eval_profile,metrics=metrics,sha256=digest(metrics)))
    return dict(format=spec,student_source_sha256=identity['student_source_sha256'],cache_key=key,training_cache_hit=trained_cache,metrics_cache_hit=metrics_cache,training=info,metrics=metrics)
