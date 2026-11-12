"""Training worker: fixed course model with the submitted Transformer functions."""
import argparse,json,traceback
from pathlib import Path
from .format_spec import read_format
from .format_cache import run_experiment
from .transformer_checks import StudentImplementationError

def main():
    p=argparse.ArgumentParser();p.add_argument('--format',required=True);p.add_argument('--profile',required=True)
    p.add_argument('--submission',required=True);p.add_argument('--cache',required=True);p.add_argument('--result',required=True);a=p.parse_args()
    try:
        spec=read_format(a.format);profile=json.loads(Path(a.profile).read_text())
        result=run_experiment(spec,a.cache,a.submission,samples=profile['tf_samples'],seed=profile['seed']+300,
                              split='test' if profile['official'] else 'validation')
        result['ok']=True
    except StudentImplementationError as error:result=dict(ok=False,student_error=True,fatal_error=str(error))
    except Exception as error:result=dict(ok=False,fatal_error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    Path(a.result).write_text(json.dumps(result,indent=2));return 0 if result['ok'] else 1
if __name__=='__main__':raise SystemExit(main())
