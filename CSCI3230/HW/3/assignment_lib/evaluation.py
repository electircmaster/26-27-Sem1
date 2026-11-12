"""Single evaluator used by both public checks and TA batch grading.

Process time limits are handled by the launcher. This module is NOT a security sandbox.
"""
import contextlib
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .loading import load_model, construct_model
from .data import fashion_dataset, make_token_batch
from .training import classification_accuracy
from .spec import PAD, SEP
from .spec import VOCAB_SIZE, TF_CONFIG
from .transformer_checks import basic_attention_only

REFERENCE_SDPA=nn.functional.scaled_dot_product_attention



def finite_logits(y,shape):
    assert isinstance(y,torch.Tensor), 'forward must return one Tensor of logits.'
    assert tuple(y.shape)==shape, f'Expected logits {shape}, got {tuple(y.shape)}.'
    assert y.is_floating_point() and torch.isfinite(y).all(), 'Logits must be finite floating-point values.'


def assert_invariant_logits(actual,reference):
    torch.testing.assert_close(actual,reference,rtol=2e-4,atol=2e-5,
                               msg=f'Logits changed after circular translation (max abs error {float((actual-reference).abs().max()):.6g}).')
    assert torch.equal(actual.argmax(-1),reference.argmax(-1)), 'Predicted classes changed after translation.'
    # Prevent multiplying logits by a tiny scalar from bypassing an absolute tolerance.
    def normalize(y):
        centered=y.double()-y.double().mean(-1,keepdim=True)
        norm=torch.linalg.vector_norm(centered,dim=-1,keepdim=True)
        return centered/norm.clamp_min(torch.finfo(torch.float64).tiny)
    torch.testing.assert_close(normalize(actual),normalize(reference),rtol=2e-4,atol=2e-5,
                               msg='Centered, normalized logits changed; rescaling logits cannot establish invariance.')


def attention_case(module,seed,dtype,masked):
    g=torch.Generator().manual_seed(seed)
    b,h,t,d=2,3,7,5
    q,k,v=[torch.randn(b,h,t,d,generator=g,dtype=dtype) for _ in range(3)]
    allowed=None
    if masked:
        allowed=torch.rand(t,t,generator=g)>.4
        allowed.fill_diagonal_(True)
    with basic_attention_only(): context,weights=module.scaled_dot_product_attention(q,k,v,allowed)
    expected=REFERENCE_SDPA(q,k,v,attn_mask=allowed,dropout_p=0.)
    identity=torch.eye(t,dtype=dtype).expand(b,h,t,t)
    expected_weights=REFERENCE_SDPA(q,k,identity,attn_mask=allowed,dropout_p=0.)
    return context,weights,expected,expected_weights,allowed


def score_accuracy(accuracy,low,high,points):
    if not 0<=low<high<=1: raise ValueError('Invalid score calibration.')
    return points*max(0.,min(1.,(accuracy-low)/(high-low)))


def evaluate(directory,profile,interfaces_only=False,data_dir=None):
    started=time.perf_counter();torch.set_num_threads(2)
    tests=[];models={};modules={};seed=int(profile['seed'])
    def run(name,points,fn):
        begin=time.perf_counter()
        try:
            detail=fn() or {}
            entry=dict(name=name,status='PASS',points=points,max_points=points,detail=detail)
        except Exception as error:
            entry=dict(name=name,status='FAIL',points=0,max_points=points,
                       detail=dict(error=f'{type(error).__name__}: {str(error)[:1500]}'))
        entry['seconds']=round(time.perf_counter()-begin,4);tests.append(entry)
        return entry['status']=='PASS'
    def skip(name,points,reason):
        tests.append(dict(name=name,status='SKIP',points=None,max_points=points,detail=dict(reason=reason)))
    for task in ['cnn','transformer']:
        model_key=task
        def interface(task=task,model_key=model_key):
            torch.manual_seed(seed)
            module,model=load_model(directory,task) if task=='cnn' else construct_model(directory,task)
            model.eval()
            with torch.inference_mode(),basic_attention_only():
                if task=='cnn':
                    for b in [1,7]:
                        x=torch.rand(b,1,28,28)*2-1
                        y=model(x);finite_logits(y,(b,10))
                        torch.testing.assert_close(y,model(x),rtol=0,atol=0,msg='eval forward must be deterministic.')
                        torch.testing.assert_close(y,torch.cat([model(row[None]) for row in x]),rtol=2e-4,atol=2e-5,
                                                   msg='Predictions must not depend on other examples in the batch.')
                else:
                    for b,t in [(1,1),(3,9),(7,TF_CONFIG['max_length'])]:
                        x=torch.randint(VOCAB_SIZE,(b,t));y=model(x);finite_logits(y,(b,t,VOCAB_SIZE))
                        torch.testing.assert_close(y,model(x),rtol=0,atol=0)
            models[model_key]=model;modules[task]=module
            return dict(parameters=sum(p.numel() for p in model.parameters()))
        run(f'{model_key}.interface',5 if task=='cnn' else 0,interface)
    if not interfaces_only:
        invariant_results=[]
        for case in range(4):
            def invariance(case=case):
                if 'cnn' not in models: raise ValueError('CNN interface must pass first.')
                torch.manual_seed(seed+case+10)
                model=models['cnn'] if case<2 else construct_model(directory,'cnn')[1].eval()
                g=torch.Generator().manual_seed(seed+case+100)
                x=torch.rand(5,1,28,28,generator=g)*2-1
                # The real-image fixture is shipped publicly; official runs use test images here.
                if case==1:
                    if data_dir:
                        split='test' if profile['official'] else 'validation'
                        x=fashion_dataset(data_dir,split,False,limit=5).tensors[0]
                    else:
                        with np.load(Path(__file__).parent/'public_fashion_examples.npz') as f:
                            x=torch.from_numpy(f['images'][:5].copy())
                largest=0.;agreement=[]
                with torch.inference_mode():
                    base=model(x)
                    for trial in range(profile['shift_trials']):
                        shifts=torch.randint(-5,6,(len(x),2),generator=g)
                        if trial==0: shifts[:]=torch.tensor([1,0])
                        elif trial==1: shifts[:]=torch.tensor([0,1])
                        else: shifts[0]=torch.tensor([-5,5])
                        moved=torch.stack([torch.roll(row,(int(s[0]),int(s[1])),dims=(-2,-1)) for row,s in zip(x,shifts)])
                        actual=model(moved)
                        largest=max(largest,float((actual-base).abs().max()))
                        agreement.append(float((actual.argmax(-1)==base.argmax(-1)).float().mean()))
                        assert_invariant_logits(actual,base)
                return dict(max_abs_logit_error=largest,prediction_agreement=min(agreement),
                            weights='submitted' if case<2 else 'fresh initialization')
            invariant_results.append(run(f'cnn.invariance_{case+1}',5,invariance))
        if 'cnn' in models and data_dir:
            def accuracy_cnn():
                split='test' if profile['official'] else 'validation'
                dataset=fashion_dataset(data_dir,split,False,limit=profile['cnn_samples'])
                acc=classification_accuracy(models['cnn'],DataLoader(dataset,batch_size=128),'cpu',shift_seed=seed+500)
                return dict(accuracy=acc,samples=len(dataset),split=split,translated=True,eligible=all(invariant_results))
            if run('cnn.accuracy',25,accuracy_cnn):
                entry=tests[-1];a=entry['detail']['accuracy']
                entry['points']=score_accuracy(a,profile['cnn_floor'],profile['cnn_full'],25) if all(invariant_results) else 0.
        elif 'cnn' not in models:
            run('cnn.accuracy',25,lambda: (_ for _ in ()).throw(ValueError('CNN interface failed.')))
        else: skip('cnn.accuracy',25,'Supply --data-dir with cached Fashion-MNIST to measure validation accuracy.')
        def get_tf():
            if 'transformer' not in modules:
                # Tensor-function credit does not depend on having a loadable trained checkpoint.
                modules['transformer']=construct_model(directory,'transformer')[0]
            return modules['transformer']
        def attention_output():
            module=get_tf()
            for dtype in [torch.float32,torch.float64]:
                for masked in [False,True]:
                    context,_,expected,_,_=attention_case(module,seed+200,dtype,masked)
                    torch.testing.assert_close(context,expected,rtol=1e-5,atol=1e-6)
            return dict(cases=4)
        def attention_weights():
            module=get_tf()
            for masked in [False,True]:
                _,w,_,expected,allowed=attention_case(module,seed+201,torch.float64,masked)
                torch.testing.assert_close(w,expected,rtol=1e-6,atol=1e-8)
                torch.testing.assert_close(w.sum(-1),torch.ones_like(w.sum(-1)),rtol=1e-6,atol=1e-8)
                if allowed is not None: assert not w[...,~allowed].any(), 'Forbidden attention weights must be zero.'
        def attention_gradient():
            module=get_tf();g=torch.Generator().manual_seed(seed+202)
            q,k,v=[torch.randn(2,2,5,3,dtype=torch.float64,generator=g,requires_grad=True) for _ in range(3)]
            mask=torch.ones(5,5,dtype=torch.bool).tril()
            with basic_attention_only(): result,_=module.scaled_dot_product_attention(q,k,v,mask)
            probe=torch.randn(result.shape,dtype=torch.float64,generator=g)
            grads=torch.autograd.grad((result*probe).sum(),(q,k,v),allow_unused=True)
            expected=REFERENCE_SDPA(q,k,v,attn_mask=mask)
            correct=torch.autograd.grad((expected*probe).sum(),(q,k,v))
            for actual,reference in zip(grads,correct):
                assert actual is not None, 'Attention must preserve gradient paths through Q, K and V.'
                torch.testing.assert_close(actual,reference,rtol=1e-5,atol=1e-7)
        def mask_test():
            module=get_tf()
            for t in [1,2,7,22,48]:
                mask=module.make_causal_mask(t,torch.device('cpu'))
                assert mask.dtype==torch.bool and tuple(mask.shape)==(t,t)
                assert torch.equal(mask,torch.arange(t)[None,:]<=torch.arange(t)[:,None]), 'True means an allowed key.'
        def targets_test():
            module=get_tf()
            for record in [['value'],['index','value','bar'],['dot','value','key','bar']]:
                fmt={'version':1,'record':record}
                tokens=make_token_batch(7,np.random.default_rng(seed+203),response_format=fmt)
                inputs,targets,mask=module.prepare_inputs_and_targets(tokens.clone())
                assert inputs.dtype==targets.dtype==torch.long and mask.dtype==torch.bool
                assert torch.equal(inputs,tokens[:,:-1]), 'Input must exclude the final column.'
                assert torch.equal(targets,tokens[:,1:]), 'Targets must be shifted one token forward.'
                expected_mask=(inputs==SEP).cumsum(1).bool() & (targets!=PAD)
                assert torch.equal(mask,expected_mask), 'Score every response token and EOS; exclude prompt and padding.'
        def future_test():
            get_tf();torch.manual_seed(seed+204);model=models['transformer']
            x=make_token_batch(3,np.random.default_rng(seed+204));cut=5
            changed=x.clone();changed[:,cut:]=torch.randint(VOCAB_SIZE,changed[:,cut:].shape)
            with torch.inference_mode(),basic_attention_only():a=model(x);b=model(changed)
            torch.testing.assert_close(a[:,:cut],b[:,:cut],rtol=2e-4,atol=2e-5,
                                      msg='Changing future tokens changed prefix predictions.')
        tf_results=[]
        for name,points,fn in [('attention_output',4,attention_output),('attention_weights',3,attention_weights),
                               ('attention_gradient',5,attention_gradient),('causal_mask',4,mask_test),
                               ('target_alignment',4,targets_test),('future_independence',0,future_test)]:
            tf_results.append(run('transformer.'+name,points,fn))
    return dict(schema='csci3230_a3_result_v5',official=profile['official'],interfaces_only=interfaces_only,
                tests=tests,score=round(sum(t['points'] or 0 for t in tests),4),
                points_evaluated=sum(t['max_points'] for t in tests if t['status']!='SKIP'),
                seconds=round(time.perf_counter()-started,4),
                ok=all(t['status']!='FAIL' for t in tests))
