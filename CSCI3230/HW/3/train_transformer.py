"""Train the fixed course model using your functions and output format."""
import argparse,json
from pathlib import Path
from assignment_lib.format_spec import read_format,BASELINE
from assignment_lib.format_cache import run_experiment
from assignment_lib.transformer_checks import StudentImplementationError

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--submission',default='submission');p.add_argument('--format')
    p.add_argument('--cache-dir',default='.format_cache');p.add_argument('--result',default='format_result.json')
    p.add_argument('--compare-baseline',action='store_true');a=p.parse_args()
    spec=read_format(a.format or Path(a.submission)/'format.json')
    try:
        result=run_experiment(spec,a.cache_dir,a.submission)
        print('Format:',spec['record']);print('Training:',result['training']);print('Validation:',result['metrics'])
        print('Cache:',dict(training=result['training_cache_hit'],metrics=result['metrics_cache_hit']))
        if a.compare_baseline:
            baseline=run_experiment(BASELINE,a.cache_dir,a.submission);result['baseline']=baseline
            print('Baseline strict accuracy:',baseline['metrics']['accuracy'])
        Path(a.result).write_text(json.dumps(result,indent=2));print('Saved',a.result)
    except StudentImplementationError as error:p.exit(1,f'Fix transformer.py before training: {error}\n')
