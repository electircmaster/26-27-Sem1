import argparse
import json
import time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from .data import fashion_dataset
from .loading import construct_model


@torch.inference_mode()
def classification_accuracy(model,loader,device,shift_seed=None):
    model.eval();correct=total=0
    generator=torch.Generator().manual_seed(shift_seed) if shift_seed is not None else None
    for x,y in loader:
        if generator is not None:
            shifts=torch.randint(-5,6,(len(x),2),generator=generator)
            x=torch.stack([torch.roll(row,(int(s[0]),int(s[1])),dims=(-2,-1)) for row,s in zip(x,shifts)])
        pred=model(x.to(device)).argmax(-1).cpu()
        correct+=int((pred==y).sum());total+=len(y)
    return correct/total



def main(task):
    if task!='cnn':raise ValueError('Use train_transformer.py for format experiments.')
    p=argparse.ArgumentParser()
    p.add_argument('--submission',default='submission')
    p.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu')
    p.add_argument('--data-dir',default='data');p.add_argument('--download',action='store_true')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--smoke',action='store_true',help='Two updates only; not a graded training run.')
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    if not 1<=args.epochs<=10:
        p.error('CNN budget: at most 10 epochs.')
    directory=Path(args.submission);destination=directory/f'{task}_weights.pt'
    if destination.exists() and not args.overwrite: p.error('Weights exist; pass --overwrite to replace them.')
    torch.set_num_threads(2);torch.manual_seed(3230)
    module,model=construct_model(directory,task);model.to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    history=[];started=time.perf_counter();best_accuracy=-1.;best_state=None;selected_epoch=None
    if task=='cnn':
        train=fashion_dataset(args.data_dir,'train',args.download,limit=256 if args.smoke else None)
        val=fashion_dataset(args.data_dir,'validation',False,limit=128 if args.smoke else None)
        loader=DataLoader(train,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(3230))
        val_loader=DataLoader(val,batch_size=128)
        for epoch in range(1,(1 if args.smoke else args.epochs)+1):
            model.train();total=0.;count=0
            for x,y in loader:
                x,y=x.to(args.device),y.to(args.device)
                loss=nn.functional.cross_entropy(model(x),y)
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                total+=float(loss.detach())*len(y);count+=len(y)
            acc=classification_accuracy(model,val_loader,args.device)
            if acc>best_accuracy:
                best_accuracy=acc;selected_epoch=epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            history.append(dict(epoch=epoch,train_loss=total/count,validation_accuracy=acc))
            print(history[-1],flush=True)
    state=best_state if task=='cnn' else {k:v.detach().cpu() for k,v in model.state_dict().items()}
    torch.save(state,destination)
    log=dict(task=task,seed=3230,device=args.device,smoke=args.smoke,history=history,
             elapsed_seconds=time.perf_counter()-started,selected_epoch=selected_epoch,
             checkpoint_selection='best validation accuracy' if task=='cnn' else 'last update')
    (directory/f'{task}_training.json').write_text(json.dumps(log,indent=2))
    print('Saved',destination,flush=True)
