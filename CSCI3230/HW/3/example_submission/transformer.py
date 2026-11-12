"""Complete the three TODO functions. The placeholders have valid shapes but are wrong.

The course model calls these functions during real training and generation.
Implement attention with basic tensor operations, not a built-in attention layer/API.
"""
import math
import torch
from assignment_lib.spec import PAD, SEP


def scaled_dot_product_attention(q,k,v,allowed_mask=None):
    """q,k,v: float [B,H,T,D]; return context [B,H,T,D], weights [B,H,T,T].

    allowed_mask: bool [T,T], True means ALLOWED. None means unrestricted attention.
    Every query has at least one allowed key. No dropout is required.
    Support float32/float64, autograd and the device of q. Do not detach tensors.
    """
    # TODO: replace the identity-attention placeholder with scaled attention.
    b,h,t,d=q.shape
    weights=torch.eye(t,dtype=q.dtype,device=q.device).expand(b,h,t,t)
    return weights@v,weights


def make_causal_mask(sequence_length,device):
    """Return bool [T,T]; a query may attend to itself and earlier positions."""
    # TODO: this all-allowed mask is an intentionally incorrect placeholder.
    return torch.ones(sequence_length,sequence_length,dtype=torch.bool,device=device)


def prepare_inputs_and_targets(tokens):
    """Complete padded tokens [B,L] -> inputs, targets, loss_mask, all [B,L-1].

    inputs/targets: int64, next-token alignment. loss_mask: bool, True only at
    positions predicting all response tokens AND EOS, after the SEP boundary.
    Context, query keys and right padding do not contribute to the loss.
    """
    # TODO: the target below is intentionally NOT shifted correctly.
    inputs=tokens[:,:-1]
    targets=tokens[:,:-1]
    mask=(inputs==SEP).cumsum(1).bool() & (tokens[:,1:]!=PAD)
    return inputs,targets,mask
