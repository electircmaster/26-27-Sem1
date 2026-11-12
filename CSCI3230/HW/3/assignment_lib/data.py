from pathlib import Path
import hashlib
import numpy as np
import torch
from .spec import PAD, BOS, QUERY, SEP, EOS, KEY_START, VALUE_START, NUM_KEYS, NUM_VALUES
from .format_spec import encode_answer, max_new_tokens, BASELINE


def make_examples(batch_size,rng,min_pairs=2,max_pairs=8):
    """Raw dictionaries and query lists, independent of response representation."""
    if not 2<=min_pairs<=max_pairs<=8:raise ValueError('Dictionary size must be within 2..8.')
    rows=[]
    for _ in range(batch_size):
        n=int(rng.integers(min_pairs,max_pairs+1))
        keys=rng.choice(NUM_KEYS,size=n,replace=False)+KEY_START
        values=rng.integers(NUM_VALUES,size=n)+VALUE_START
        count=int(rng.integers(2,min(4,n)+1));indices=rng.choice(n,size=count,replace=False)
        pairs=np.stack([keys,values],axis=1).reshape(-1).tolist()
        prompt=[BOS]+pairs+[QUERY]+keys[indices].tolist()+[SEP]
        rows.append(dict(prompt=prompt,query_keys=keys[indices].tolist(),values=values[indices].tolist()))
    return rows


def example_bytes(example):
    # Prompt includes the entire dictionary and ordered queries, but no answer encoding.
    return np.asarray(example['prompt'],dtype='<i8').tobytes()


def split_examples(batch_size,rng,split,min_pairs=2,max_pairs=8):
    buckets={'train':set(range(8)),'validation':{8},'test':{9}}
    if split not in buckets:raise ValueError('Unknown split.')
    rows=[]
    while len(rows)<batch_size:
        for row in make_examples(max(32,2*(batch_size-len(rows))),rng,min_pairs,max_pairs):
            if hashlib.sha256(example_bytes(row)).digest()[0]%10 in buckets[split]:rows.append(row)
            if len(rows)==batch_size:break
    return rows


def pad_rows(rows):
    result=torch.full((len(rows),max(map(len,rows))),PAD,dtype=torch.long)
    for i,row in enumerate(rows):result[i,:len(row)]=torch.tensor(row)
    return result


def encode_examples(examples,response_format=None):
    response_format=BASELINE if response_format is None else response_format
    return pad_rows([e['prompt']+encode_answer(e['query_keys'],e['values'],response_format) for e in examples])


def make_token_batch(batch_size,rng,min_pairs=2,max_pairs=8,response_format=None):
    return encode_examples(make_examples(batch_size,rng,min_pairs,max_pairs),response_format)


def split_token_batch(batch_size,rng,split,min_pairs=2,max_pairs=8,response_format=None):
    return encode_examples(split_examples(batch_size,rng,split,min_pairs,max_pairs),response_format)


def make_queries(batch_size,rng,min_pairs=2,max_pairs=8,split='validation',response_format=None):
    response_format=BASELINE if response_format is None else response_format
    examples=split_examples(batch_size,rng,split,min_pairs,max_pairs)
    prefixes=pad_rows([e['prompt'] for e in examples]);lengths=torch.tensor([len(e['prompt']) for e in examples])
    targets=torch.full((batch_size,max_new_tokens(response_format)),PAD,dtype=torch.long)
    for i,e in enumerate(examples):
        answer=encode_answer(e['query_keys'],e['values'],response_format)
        targets[i,:len(answer)]=torch.tensor(answer)
    return prefixes,lengths,targets


def format_tokens(tokens):
    from .spec import INDEX_START,BAR,DOT
    names = {PAD:'<PAD>', BOS:'<BOS>', QUERY:'<QUERY>', SEP:'<SEP>', EOS:'<EOS>',BAR:'|',DOT:'.'}
    names.update({INDEX_START+i:f'I{i+1}' for i in range(4)})
    return ' '.join(names.get(int(t), f'K{int(t)-KEY_START}' if int(t)<VALUE_START
                              else f'V{int(t)-VALUE_START}') for t in tokens if int(t)!=PAD)


def fashion_dataset(directory, split, download=False, limit=None):
    """Only the course-owned TA evaluator requests split='test'."""
    from torchvision.datasets import FashionMNIST
    from torch.utils.data import TensorDataset
    if split not in {'train','validation','test'}:
        raise ValueError('Choose train, validation or test.')
    dataset = FashionMNIST(str(Path(directory)), train=(split!='test'), download=download)
    if split == 'test':
        ids = torch.arange(len(dataset))
    else:
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(3230))
        ids = order[:50_000] if split=='train' else order[50_000:]
    if limit is not None:
        ids = ids[:limit]
    # Exactly the same normalization is used for training and evaluation.
    images = dataset.data[ids, None].float().div(127.5).sub(1)
    return TensorDataset(images, dataset.targets[ids].long())
