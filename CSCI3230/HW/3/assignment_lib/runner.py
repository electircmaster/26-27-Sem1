"""Coordinator: student model training and correctness checks have separate process budgets."""
import json,math,os,signal,subprocess,sys,tempfile,time
from pathlib import Path
from .format_spec import read_format,canonical_format
ROOT=Path(__file__).resolve().parents[1]


def _worker(module,args,output,log_path,timeout):
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONHASHSEED='0')
    with log_path.open('w+') as log:
        process=subprocess.Popen([sys.executable,'-m',module]+args+['--result',str(output)],cwd=ROOT,env=env,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name=='posix':os.killpg(process.pid,signal.SIGKILL)
            else:process.kill()
            process.wait();result=dict(ok=False,fatal_error=f'Exceeded {timeout:.1f}s time limit.',timed_out=True,timeout_seconds=timeout,worker_module=module)
        else:
            try:result=json.loads(output.read_text())
            except (OSError,ValueError):result=dict(ok=False,fatal_error=f'Worker exited {process.returncode} without a valid report.')
        log.seek(0,2);size=log.tell();log.seek(max(0,size-8000));result['worker_log_tail']=log.read()
    return result


def acknowledgement_check(submission):
    try:
        data=json.loads((Path(submission)/'acknowledgement.json').read_text())
        if not isinstance(data,dict):raise ValueError('Expected a JSON object.')
        missing=[key for key in ['cnn','transformer'] if data.get(key)!='yes']
        if missing:raise ValueError('These fields must contain the exact string "yes": '+', '.join(missing))
    except (OSError,ValueError,UnicodeError) as error:
        return dict(name='submission.acknowledgement',status='FAIL',points=0,max_points=0,
                    detail=dict(error=str(error),consequence='Zero for the entire assignment, including PDF analysis.'))
    return dict(name='submission.acknowledgement',status='PASS',points=0,max_points=0,detail={})


def run_checks(submission,profile,result_path,data_dir=None,interfaces_only=False,timeout=300,cache_dir=None,format_timeout=300):
    # Training (including cache waits) cannot consume the correctness worker's budget.
    if any(not math.isfinite(t) or t<=0 for t in [timeout,format_timeout]):
        raise ValueError('Correctness and format timeouts must be positive finite seconds.')
    started=time.perf_counter();submission=Path(submission).resolve();profile=Path(profile).resolve()
    result_path=Path(result_path).resolve();result_path.parent.mkdir(parents=True,exist_ok=True)
    settings=json.loads(profile.read_text());format_test=None;experiment=None
    acknowledgement=acknowledgement_check(submission) if settings.get('official') else None
    if acknowledgement and acknowledgement['status']=='FAIL':
        result=dict(schema='csci3230_a3_result_v5',official=True,interfaces_only=interfaces_only,
                    tests=[acknowledgement],score=0,points_evaluated=0,assignment_zero=True,final_score=0,
                    analysis_points=0,ok=False,needs_retry=False,automated_max_points=95,manual_analysis_max_points=5,
                    seconds=round(time.perf_counter()-started,4),
                    timeouts=dict(correctness_seconds=timeout,format_seconds=format_timeout))
        result_path.write_text(json.dumps(result,indent=2));return result
    cache=Path(cache_dir).resolve() if cache_dir else ROOT/'.format_cache'
    with tempfile.TemporaryDirectory(prefix='a3_check_') as tmp:
        tmp=Path(tmp)
        try:
            spec=read_format(submission/'format.json')
            format_test=dict(name='transformer.format',status='PASS',points=5,max_points=5,detail=spec)
            (tmp/'format.json').write_text(canonical_format(spec))
        except Exception as error:
            format_test=dict(name='transformer.format',status='FAIL',points=0,max_points=5,detail=dict(error=f'{type(error).__name__}: {error}'))
        # Train the fixed architecture with submitted functions; keep a separate correctness budget.
        if not interfaces_only and format_test['status']=='PASS':
            experiment=_worker('assignment_lib.format_worker',['--submission',str(submission),'--format',str(tmp/'format.json'),'--profile',str(profile),'--cache',str(cache)],
                               tmp/'format_result.json',tmp/'format.log',format_timeout)
        args=['--submission',str(submission),'--profile',str(profile)]
        if data_dir:args+=['--data-dir',str(Path(data_dir).resolve())]
        if interfaces_only:args+=['--interfaces-only']
        result=_worker('assignment_lib.worker',args,tmp/'function_result.json',tmp/'function.log',timeout)
    if 'tests' not in result:
        result['format_result']=experiment;result['format_validation']=format_test
    else:
        if acknowledgement:result['tests'].append(acknowledgement)
        result['tests'].append(format_test)
        if not interfaces_only:
            if experiment and experiment.get('ok'):
                acc=experiment['metrics']['accuracy'];lo,hi=settings['tf_floor'],settings['tf_full']
                points=20*max(0.,min(1.,(acc-lo)/(hi-lo)))
                detail={k:v for k,v in experiment.items() if k not in ['worker_log_tail','ok']}
                result['tests'].append(dict(name='transformer.performance',status='PASS',points=points,max_points=20,detail=detail))
            elif experiment and experiment.get('student_error'):
                result['tests'].append(dict(name='transformer.performance',status='FAIL',points=0,max_points=20,
                    detail=dict(error=experiment['fatal_error'],student_error=True)))
            elif format_test['status']=='PASS':
                # A timeout/infrastructure failure is unresolved; implementation errors above earn zero performance points.
                detail=dict(error=(experiment or {}).get('fatal_error','Format worker returned no result.'),
                            timed_out=bool((experiment or {}).get('timed_out')),retryable=True,
                            timeout_seconds=format_timeout,worker_log_tail=(experiment or {}).get('worker_log_tail',''))
                result['tests'].append(dict(name='transformer.performance',status='ERROR',points=None,max_points=20,detail=detail))
            else:
                result['tests'].append(dict(name='transformer.performance',status='FAIL',points=0,max_points=20,
                                           detail=dict(error='Invalid format specification.')))
        result['score']=round(sum(t['points'] or 0 for t in result['tests']),4)
        result['points_evaluated']=sum(t['max_points'] for t in result['tests'] if t['status'] not in ['SKIP','ERROR'])
        result['needs_retry']=any(t['status']=='ERROR' for t in result['tests'])
        result['ok']=all(t['status'] not in ['FAIL','ERROR'] for t in result['tests'])
        result['seconds']=round(time.perf_counter()-started,4)
    result['automated_max_points']=95
    result['manual_analysis_max_points']=5
    result['timeouts']=dict(correctness_seconds=timeout,format_seconds=format_timeout)
    result_path.write_text(json.dumps(result,indent=2));return result


def print_result(result):
    if 'fatal_error' in result:print('[ERROR]',result['fatal_error'])
    for test in result.get('tests',[]):
        detail=test['detail']
        if test['name']=='transformer.performance' and 'metrics' in detail:
            detail=dict(format=detail['format']['record'],accuracy=detail['metrics']['accuracy'],
                        value_token_accuracy=detail['metrics']['value_token_accuracy'],cache_hit=detail['metrics_cache_hit'])
        print(f"[{test['status']}] {test['name']}  {test['points']}/{test['max_points']}  {detail}")
    if result.get('assignment_zero'):
        print('Final assignment score: 0/100. Required acknowledgement missing or invalid; PDF points cannot be added.')
        return
    if not result.get('interfaces_only') and 'score' in result:
        note='' if result.get('official') else ' Public results are not final grades.'
        label='Partial score (format experiment needs retry)' if result.get('needs_retry') else 'Automated score'
        print(f"{label}: {result['score']:.2f}/{result['points_evaluated']} evaluated points. PDF analysis: 5 points awarded separately by the TA.{note}")
