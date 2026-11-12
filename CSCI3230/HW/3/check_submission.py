import argparse
from pathlib import Path
from assignment_lib.runner import ROOT,run_checks,print_result

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Public checks using the official shared evaluator in a fresh process.')
    p.add_argument('--submission',default='submission')
    p.add_argument('--interfaces-only',action='store_true')
    p.add_argument('--cache-dir');p.add_argument('--data-dir');p.add_argument('--result',default='check_result.json')
    p.add_argument('--timeout',type=int,default=300,help='Independent correctness-worker limit in seconds.')
    p.add_argument('--format-timeout',type=int,default=300,help='Independent format experiment limit, including cache waits.')
    args=p.parse_args()
    if args.timeout<=0 or args.format_timeout<=0:p.error('Both timeouts must be positive.')
    result=run_checks(args.submission,ROOT/'public_profile.json',args.result,args.data_dir,args.interfaces_only,args.timeout,args.cache_dir,args.format_timeout)
    print_result(result);print('Report:',Path(args.result).resolve())
    raise SystemExit(0 if result['ok'] else 1)
