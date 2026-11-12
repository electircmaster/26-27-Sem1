"""Small, actionable pre-training checks, not a general-purpose code sandbox."""
import contextlib
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from .spec import PAD,SEP,VOCAB_SIZE
from .data import make_token_batch
from .loading import transformer_from_functions

class StudentImplementationError(ValueError):
    pass

@contextlib.contextmanager
def basic_attention_only():
    message='Implement attention with basic tensor operations; built-in attention is not permitted.'
    with patch('torch.nn.functional.scaled_dot_product_attention',side_effect=RuntimeError(message)), \
         patch.object(nn.MultiheadAttention,'forward',side_effect=RuntimeError(message)):
        yield



def checked_targets(module,tokens):
    try:
        inputs,targets,mask=module.prepare_inputs_and_targets(tokens.clone())
        if not all(isinstance(x,torch.Tensor) for x in [inputs,targets,mask]):raise ValueError('Return three tensors.')
        if inputs.dtype!=torch.long or targets.dtype!=torch.long or mask.dtype!=torch.bool:raise ValueError('Use int64 inputs/targets and bool loss_mask.')
        if not torch.equal(inputs,tokens[:,:-1]) or not torch.equal(targets,tokens[:,1:]):raise ValueError('Shift targets exactly one token forward.')
        expected=(tokens[:,:-1]==SEP).cumsum(1).bool() & (tokens[:,1:]!=PAD)
        if not torch.equal(mask,expected):raise ValueError('Score every response token and EOS, excluding prompt/padding.')
        return inputs,targets,mask
    except Exception as error:
        raise StudentImplementationError(f'prepare_inputs_and_targets: {error}') from error


def preflight(module,spec):
    """Check trainability and no target leakage, without requiring perfect numeric attention."""
    try:
        with torch.random.fork_rng(),basic_attention_only():
            torch.manual_seed(71)
            for t in [1,7,48]:
                mask=module.make_causal_mask(t,torch.device('cpu'))
                if not isinstance(mask,torch.Tensor) or mask.dtype!=torch.bool or mask.device.type!='cpu' or not torch.equal(mask,torch.ones(t,t,dtype=torch.bool).tril()):
                    raise ValueError('make_causal_mask must return the lower-triangular bool mask, True=allowed.')
            q,k,v=[torch.randn(2,2,7,4,requires_grad=True) for _ in range(3)]
            context,_=module.scaled_dot_product_attention(q,k,v,torch.ones(7,7,dtype=torch.bool).tril())
            if not isinstance(context,torch.Tensor) or context.shape!=q.shape or context.dtype!=q.dtype or not torch.isfinite(context).all():
                raise ValueError('Attention context must have shape [B,H,T,D] and finite floating values.')
            gradients=torch.autograd.grad((context*torch.randn_like(context)).sum(),(q,k,v),allow_unused=True)
            if any(g is None or not torch.isfinite(g).all() for g in gradients):raise ValueError('Attention must preserve finite gradient paths through Q, K and V.')
            tokens=make_token_batch(7,np.random.default_rng(71),response_format=spec)
            checked_targets(module,tokens)
            model=transformer_from_functions(module).eval();changed=tokens.clone();cut=5
            changed[:,cut:]=torch.randint(VOCAB_SIZE,changed[:,cut:].shape)
            with torch.no_grad():a=model(tokens);b=model(changed)
            if a.shape!=(*tokens.shape,VOCAB_SIZE) or not torch.isfinite(a).all():raise ValueError('Invalid model output shape or nonfinite logits.')
            torch.testing.assert_close(a[:,:cut],b[:,:cut],rtol=2e-4,atol=2e-5,msg='Changing future tokens changed earlier predictions; check mask use in attention.')
    except StudentImplementationError:raise
    except Exception as error:raise StudentImplementationError(f'Training precheck: {error}') from error
