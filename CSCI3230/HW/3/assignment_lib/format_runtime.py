"""Fixed-budget training and generation using the submitted three functions."""
import hashlib,json,platform,time
from pathlib import Path
import numpy as np
import torch
from .spec import TF_CONFIG,TF_TRAIN_STEPS,TF_SEED,PAD,SEP
from .loading import transformer_from_functions
from .transformer_checks import basic_attention_only,checked_targets,StudentImplementationError
from .data import split_examples,encode_examples,example_bytes,make_queries
from .format_spec import validate_format,canonical_format,max_new_tokens,parse_answer


def state_digest(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode());h.update(str(tuple(value.shape)).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train_fixed(spec,module,steps=TF_TRAIN_STEPS):
    """steps override is for TA calibration; public/official entry points always fix it."""
    spec=validate_format(spec);torch.set_num_threads(2);torch.manual_seed(TF_SEED)
    model=transformer_from_functions(module);initial=state_digest(model.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    rng=np.random.default_rng(TF_SEED);stream=hashlib.sha256();start=time.perf_counter()
    input_tokens=supervised_tokens=0;losses=[]
    for step in range(1,steps+1):
        model.train();examples=split_examples(128,rng,'train')
        for e in examples:stream.update(example_bytes(e))
        tokens=encode_examples(examples,spec)
        inputs,targets,mask=checked_targets(module,tokens)
        input_tokens+=int((inputs!=PAD).sum());supervised_tokens+=int(mask.sum())
        try:
            with basic_attention_only():
                loss=torch.nn.functional.cross_entropy(model(inputs)[mask],targets[mask])
                if not torch.isfinite(loss):raise ValueError('Training loss is nonfinite.')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise ValueError('Training gradients are nonfinite.')
                optimizer.step()
        except Exception as error:raise StudentImplementationError(f'Training update {step}: {error}') from error
        if step%50==0 or step==steps:losses.append(dict(step=step,loss=float(loss.detach())))
    info=dict(steps=steps,seed=TF_SEED,data_seed=TF_SEED,batch_size=128,learning_rate=1e-3,weight_decay=1e-4,
              parameters=sum(p.numel() for p in model.parameters()),initial_state_sha256=initial,
              semantic_stream_sha256=stream.hexdigest(),input_tokens=input_tokens,supervised_tokens=supervised_tokens,
              train_seconds=time.perf_counter()-start,loss_history=losses)
    return model.eval(),info


@torch.inference_mode()
def generate_answers(model,prefixes,lengths,max_tokens):
    """Free-running full-vocabulary argmax; no targets, forced keys, or true lengths."""
    model.eval();lengths=lengths.clone();batch=len(prefixes)
    tokens=torch.full((batch,prefixes.shape[1]+max_tokens),PAD,dtype=torch.long)
    tokens[:,:prefixes.shape[1]]=prefixes
    result=torch.full((batch,max_tokens),PAD,dtype=torch.long);active=torch.ones(batch,dtype=torch.bool)
    rows=torch.arange(batch)
    from .spec import EOS
    for step in range(max_tokens):
        logits=model(tokens[:,:int(lengths.max())])
        if not torch.isfinite(logits).all():raise ValueError('Generated logits are nonfinite.')
        predicted=logits[rows,lengths-1].argmax(-1)
        result[active,step]=predicted[active]
        tokens[rows[active],lengths[active]]=predicted[active];lengths[active]+=1
        active &= predicted!=EOS
        if not active.any():break
    return result


def response_statistics(predicted,targets,spec):
    rows=[]
    for pred,target in zip(predicted,targets):
        values,valid=parse_answer(pred.tolist(),spec);expected,_=parse_answer(target.tolist(),spec)
        rows.append(dict(exact=bool(torch.equal(pred,target)),value_exact=valid and values==expected,
                         count=len(expected),valid=valid,
                         token_correct=sum(i<len(values) and values[i]==v for i,v in enumerate(expected)) if valid else 0))
    return rows


@torch.inference_mode()
def evaluate_fixed(model,spec,samples=512,seed=2026,split='validation'):
    torch.set_num_threads(2);spec=validate_format(spec);rng=np.random.default_rng(seed);rows=[];start=time.perf_counter()
    for offset in range(0,samples,128):
        prefixes,lengths,targets=make_queries(min(128,samples-offset),rng,split=split,response_format=spec)
        try:
            with basic_attention_only():predicted=generate_answers(model,prefixes,lengths,max_new_tokens(spec))
        except Exception as error:raise StudentImplementationError(f'Autoregressive generation: {error}') from error
        rows.extend(response_statistics(predicted,targets,spec))
    avg=lambda key:sum(r[key] for r in rows)/len(rows)
    return dict(accuracy=avg('exact'),strict_response_accuracy=avg('exact'),value_sequence_accuracy=avg('value_exact'),
                value_token_accuracy=sum(r['token_correct'] for r in rows)/sum(r['count'] for r in rows),
                syntax_accuracy=avg('valid'),samples=samples,seed=seed,split=split,
                by_query_count={str(n):dict(samples=sum(r['count']==n for r in rows),
                    strict_response_accuracy=sum(r['exact'] for r in rows if r['count']==n)/max(1,sum(r['count']==n for r in rows)),
                    value_sequence_accuracy=sum(r['value_exact'] for r in rows if r['count']==n)/max(1,sum(r['count']==n for r in rows))) for n in [2,3,4]},
                evaluation_seconds=time.perf_counter()-start)
