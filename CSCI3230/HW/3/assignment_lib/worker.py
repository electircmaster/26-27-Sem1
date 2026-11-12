import argparse
import json
import traceback
from pathlib import Path
from .evaluation import evaluate

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--submission',required=True);p.add_argument('--profile',required=True)
    p.add_argument('--result',required=True);p.add_argument('--data-dir')
    p.add_argument('--interfaces-only',action='store_true');args=p.parse_args()
    try:
        profile=json.loads(Path(args.profile).read_text())
        result=evaluate(Path(args.submission),profile,args.interfaces_only,args.data_dir)
    except Exception as error:
        result=dict(ok=False,fatal_error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    Path(args.result).write_text(json.dumps(result,indent=2))
    return 0 if result['ok'] else 1

if __name__=='__main__': raise SystemExit(main())
