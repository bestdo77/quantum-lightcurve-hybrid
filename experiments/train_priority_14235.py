"""Train and evaluate the five priority-14235 quantum directions."""
import argparse,copy,hashlib,json,math,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score,f1_score
from natural_source_order import source_order
from priority_14235_models import make_model
from safe_capturable_adamw import SafeCapturableAdamW

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'priority_14235_20261013'
DATA=ROOT/'hierarchical_lightgbm_tree'/'data'
PLAN=RUN/'plan.json'
ALPHAS=(0.,.25,.5,.75,1.)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()

def write(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)

def atomic_torch(payload,path):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(payload,tmp);tmp.replace(path)

def metric(y,pred):
    return {'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),'accuracy':float(accuracy_score(y,pred))}

def alpha_grid(y,logits,logweights):
    rows=[{'alpha':a,'metrics':metric(y,(logits-a*logweights).argmax(1))} for a in ALPHAS]
    return rows,max(rows,key=lambda r:(r['metrics']['macro_f1'],r['metrics']['accuracy']))

def validate_plan(plan):
    assert plan['status']=='frozen' and not plan['confirmation_accessed']
    assert tuple(plan['prior_alpha_grid'])==ALPHAS and sha(DATA/'manifest.json')==plan['data_manifest_sha256']
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,(name,digest,sha(ROOT/name))

def load_data(device):
    manifest=json.loads((DATA/'manifest.json').read_text())
    for name,digest in manifest['outputs'].items():assert sha(DATA/name)==digest,name
    ycpu=np.load(DATA/'fit_y.npy');x=torch.from_numpy(np.load(DATA/'fit_x.npy')).to(device);y=torch.from_numpy(ycpu.copy()).to(device)
    freq=np.bincount(ycpu,minlength=10);weights=torch.tensor(len(ycpu)/(10*freq),dtype=torch.float32,device=device)
    logweights=np.log(len(ycpu)/(10*freq))
    val={'x':torch.from_numpy(np.load(DATA/'val_x.npy')).to(device),'y':np.load(DATA/'val_y.npy'),'ids':np.load(DATA/'val_ids.npy')}
    return x,y,ycpu,weights,logweights,val

def optimizer(model,job,device):
    buckets={'backbone':[],'theta':[],'encoding':[],'branch':[]}
    for name,p in model.named_parameters():
        if name.startswith('backbone.'):buckets['backbone'].append(p)
        elif name.endswith('.theta'):buckets['theta'].append(p)
        elif name.endswith('.frequency') or name.endswith('.shift'):buckets['encoding'].append(p)
        else:buckets['branch'].append(p)
    freeze=job.get('freeze_backbone',False);warmup=int(job.get('backbone_warmup_epochs',0))
    cfg={
      'backbone':(0. if freeze else 5e-5,0. if freeze else 1e-5,1e-4,warmup),
      'theta':(8e-4,8e-5,1e-5,0),
      'encoding':(5e-4,5e-5,0.,0),
      'branch':(1e-3,1e-4,1e-4,0)}
    groups=[]
    for name,params in buckets.items():
        if not params:continue
        start,end,decay,warm=cfg[name]
        groups.append({'params':params,'lr':torch.tensor(0. if warm else start,device=device),'start':start,'end':end,
                       'weight_decay':decay,'warmup_epochs':warm,'group_name':name})
    opt=SafeCapturableAdamW(groups,eps=1e-8,capturable=device.type=='cuda',foreach=False)
    for p in model.parameters():opt.state[p]={'step':torch.zeros((),device=device),'exp_avg':torch.zeros_like(p),'exp_avg_sq':torch.zeros_like(p)}
    return opt

def decorrelation(features):
    z=features-features.mean(0,keepdim=True)
    z=F.normalize(z,dim=-1)
    corr=torch.einsum('brf,bsf->rs',z,z)/len(z)
    eye=torch.eye(corr.shape[0],device=corr.device)
    return ((corr-eye)**2).sum()/max(corr.numel()-corr.shape[0],1)

def loss_value(model,x,y,weights,job):
    logits,aux=model(x,return_aux=True)
    per=F.cross_entropy(logits,y,weight=weights,reduction='none')
    if job['objective']=='residual_focus':
        prob=aux['base'].detach().softmax(-1);conf=prob.max(-1).values
        focus=1.+2.*(prob.argmax(-1)!=y).float()+1.5*(1.-conf)
        loss=(per*focus).sum()/(weights[y]*focus).sum().clamp_min(1e-8)
    else:
        loss=per.sum()/weights[y].sum().clamp_min(1e-8)
    if model.variant=='heterogeneous':loss=loss+job['decorrelation_weight']*decorrelation(aux['module_features'])
    if model.variant=='sparse_experts':
        gates=aux['gates'];entropy=-(gates*gates.clamp_min(1e-8).log()).sum(-1).mean()
        balance=(gates.mean(0)-1./gates.shape[1]).square().mean()
        loss=loss+job['gate_entropy_weight']*entropy+job['gate_balance_weight']*balance
    return loss

def eager_step(model,opt,x,y,weights,job):
    opt.zero_grad(set_to_none=False);loss=loss_value(model,x,y,weights,job);loss.backward();opt.step();return loss.detach()

class GraphStep:
    def __init__(self,model,opt,x,y,weights,job):
        self.x,self.y=x.clone(),y.clone();self.model,self.opt,self.weights,self.job=model,opt,weights,job
        sm={k:v.detach().clone() for k,v in model.state_dict().items()}
        so={p:{k:v.clone() for k,v in s.items()} for p,s in opt.state.items()}
        rng=torch.cuda.get_rng_state()
        def restore():
            with torch.no_grad():
                for k,v in model.state_dict().items():v.copy_(sm[k])
                for p,values in so.items():
                    for k,v in values.items():opt.state[p][k].copy_(v)
            torch.cuda.set_rng_state(rng)
        model.train();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):eager_step(model,opt,self.x,self.y,weights,job)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();restore()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):self.loss=eager_step(model,opt,self.x,self.y,weights,job)
        restore();torch.cuda.synchronize()
    def __call__(self,x,y):
        self.x.copy_(x);self.y.copy_(y);self.graph.replay();return self.loss

@torch.no_grad()
def predict(model,x,batch=512):
    model.eval();return torch.cat([model(x[s:s+batch]).float().cpu() for s in range(0,len(x),batch)]).numpy()

def gradient_stats(model):
    rows={}
    for name,p in model.named_parameters():
        if p.grad is None or not any(key in name for key in ('theta','frequency','shift')):continue
        g=p.grad.detach().float();flat=g.reshape(-1)
        row={'rms':float(g.square().mean().sqrt()),'mean_abs':float(g.abs().mean()),'max_abs':float(g.abs().max()),
             'fraction_lt_1e-8':float((g.abs()<1e-8).float().mean()),'fraction_lt_1e-7':float((g.abs()<1e-7).float().mean()),
             'fraction_lt_1e-6':float((g.abs()<1e-6).float().mean())}
        if g.ndim>=2:row['per_layer_rms']=[float(v) for v in g.reshape(g.shape[0],-1).square().mean(1).sqrt()]
        rows[name]=row
    return rows

def warm_start(model,source):
    saved=torch.load(source,map_location='cpu',weights_only=True);target=model.state_dict();copied={}
    with torch.no_grad():
        for name,value in saved.items():
            if name not in target:continue
            if target[name].shape==value.shape:
                target[name].copy_(value);copied[name]='full'
            elif name in ('readout.theta','readout.frequency','readout.shift') and target[name].shape[1:]==value.shape[1:] and target[name].shape[0]>value.shape[0]:
                target[name][:value.shape[0]].copy_(value);copied[name]=f'prefix_{value.shape[0]}'
    model.load_state_dict(target)
    return copied

def load_split(name,device):
    return {'x':torch.from_numpy(np.load(DATA/f'{name}_x.npy')).to(device),'y':np.load(DATA/f'{name}_y.npy'),'ids':np.load(DATA/f'{name}_ids.npy')}

def evaluate(model,checkpoint,alpha,device,logweights,splits,out,label):
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));metrics={}
    for split,payload in splits.items():
        logits=predict(model,payload['x']);corrected=logits-alpha*logweights;pred=corrected.argmax(1)
        metrics[split]=metric(payload['y'],pred)
        np.savez_compressed(out/f'{split}_{label}_predictions.npz',ids=payload['ids'],y=payload['y'],pred=pred,
                            logits=logits,corrected_logits=corrected,prior_alpha=np.asarray(alpha))
    return metrics

def train(job,plan,device):
    fit_x,fit_y,fit_y_cpu,weights,logweights,val=load_data(device)
    out=RUN/'models'/job['name'];out.mkdir(parents=True,exist_ok=True)
    result_path=out/'training_result.json'
    if result_path.exists():
        result=json.loads(result_path.read_text());assert result['plan_sha256']==sha(PLAN);return result
    cfg_path,latest=out/'config.json',out/'latest.pt';existing=cfg_path.exists()
    torch.manual_seed(plan['seed']);model=make_model(job['spec']).to(device);initialization={}
    if not existing and job.get('init_from'):
        source=RUN/'models'/job['init_from']/job.get('init_checkpoint','best_macro.pt')
        assert source.exists(),source;initialization={'source':str(source),'sha256':sha(source),'copied':warm_start(model,source)}
    opt=optimizer(model,job,device)
    if not existing:
        torch.save(model.state_dict(),out/'initial.pt')
        write(cfg_path,{'job':job,'counts':model.counts(),'plan_sha256':sha(PLAN),'initialization':initialization,
                        'source_sha256':plan['source_sha256'],'data_manifest_sha256':plan['data_manifest_sha256']})
    else:
        cfg=json.loads(cfg_path.read_text());assert cfg['plan_sha256']==sha(PLAN) and latest.exists()
    history=[];start_epoch=cursor=0;partial=None;prior_seconds=0.
    if existing:
        saved=torch.load(latest,map_location=device,weights_only=False);model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        history,start_epoch,cursor,partial=saved['history'],saved['epoch'],saved.get('cursor',0),saved.get('partial_loss')
        prior_seconds=saved.get('elapsed_seconds',0.)
    if not history:
        grid,selected=alpha_grid(val['y'],predict(model,val['x']),logweights)
        row={'epoch':0,'train_loss':None,'validation':selected['metrics'],'selected_alpha':selected['alpha'],'validation_grid':grid,
             'gradient_stats':{},'checkpoint':'initialized'}
        history=[row];torch.save(model.state_dict(),out/'best_macro.pt');torch.save(model.state_dict(),out/'best_acc.pt')
    best_macro=max(history,key=lambda r:(r['validation']['macro_f1'],r['validation']['accuracy']))
    best_acc=max(history,key=lambda r:(r['validation']['accuracy'],r['validation']['macro_f1']))
    graph=GraphStep(model,opt,fit_x[:plan['batch_size']],fit_y[:plan['batch_size']],weights,job) if device.type=='cuda' else None
    started=time.perf_counter()
    if not existing:atomic_torch({'model':model.state_dict(),'optimizer':opt.state_dict(),'history':history,'epoch':0,'cursor':0,'partial_loss':torch.zeros((),device=device),'elapsed_seconds':0.},latest)
    for epoch in range(start_epoch,job['epochs']):
        for group in opt.param_groups:
            warm=int(group.get('warmup_epochs',0))
            if epoch<warm:lr=0.
            else:
                frac=(epoch-warm)/max(job['epochs']-warm-1,1);cos=.5*(1+math.cos(math.pi*frac))
                lr=group['end']+(group['start']-group['end'])*cos
            group['lr'].fill_(lr)
        order=torch.from_numpy(source_order(fit_y_cpu,epoch,plan['batch_size'],20261011,10)).to(device)
        model.train();pos=cursor if epoch==start_epoch else 0
        total=partial.clone() if epoch==start_epoch and partial is not None else torch.zeros((),device=device);snapshot=None
        for start in range(pos,len(order),plan['batch_size']):
            idx=order[start:start+plan['batch_size']];xb,yb=fit_x[idx],fit_y[idx]
            loss=graph(xb,yb) if graph is not None and len(idx)==plan['batch_size'] else eager_step(model,opt,xb,yb,weights,job)
            if snapshot is None:snapshot=gradient_stats(model)
            total+=loss*len(idx);cursor=start+len(idx)
            if (start//plan['batch_size']+1)%plan.get('checkpoint_every_batches',128)==0 and cursor<len(order):
                elapsed=prior_seconds+time.perf_counter()-started
                atomic_torch({'model':model.state_dict(),'optimizer':opt.state_dict(),'history':history,'epoch':epoch,'cursor':cursor,'partial_loss':total,'elapsed_seconds':elapsed},latest)
                write(out/'progress.json',{'status':'training','epoch':epoch+1,'cursor':cursor,'samples_per_epoch':len(order),'elapsed_seconds':elapsed})
        ce=float(total/len(fit_y));assert math.isfinite(ce)
        grid,selected=alpha_grid(val['y'],predict(model,val['x']),logweights)
        row={'epoch':epoch+1,'train_loss':ce,'validation':selected['metrics'],'selected_alpha':selected['alpha'],
             'validation_grid':grid,'learning_rates':{g['group_name']:float(g['lr']) for g in opt.param_groups},'gradient_stats':snapshot}
        history.append(row)
        if (row['validation']['macro_f1'],row['validation']['accuracy'])>(best_macro['validation']['macro_f1'],best_macro['validation']['accuracy']):
            best_macro=row;torch.save(model.state_dict(),out/'best_macro.pt')
        if (row['validation']['accuracy'],row['validation']['macro_f1'])>(best_acc['validation']['accuracy'],best_acc['validation']['macro_f1']):
            best_acc=row;torch.save(model.state_dict(),out/'best_acc.pt')
        elapsed=prior_seconds+time.perf_counter()-started
        atomic_torch({'model':model.state_dict(),'optimizer':opt.state_dict(),'history':history,'epoch':epoch+1,'cursor':0,'partial_loss':torch.zeros((),device=device),'elapsed_seconds':elapsed},latest)
        cursor,partial=0,None
        write(out/'progress.json',{'status':'training',**row,'best_macro_epoch':best_macro['epoch'],'best_acc_epoch':best_acc['epoch'],'elapsed_seconds':elapsed})
        print(json.dumps({'model':job['name'],'epoch':epoch+1,'val':row['validation'],'best_macro':best_macro['validation'],'best_acc':best_acc['validation']}),flush=True)
    splits={'validation':val,'report':load_split('report',device),'extra':load_split('extra',device)}
    metrics_macro=evaluate(model,out/'best_macro.pt',best_macro['selected_alpha'],device,logweights,splits,out,'best_macro')
    metrics_acc=evaluate(model,out/'best_acc.pt',best_acc['selected_alpha'],device,logweights,splits,out,'best_acc')
    result={'name':job['name'],'priority_direction':job['priority_direction'],'job':job,'counts':model.counts(),'history':history,
            'best_macro':{'epoch':best_macro['epoch'],'alpha':best_macro['selected_alpha'],'metrics':metrics_macro},
            'best_acc':{'epoch':best_acc['epoch'],'alpha':best_acc['selected_alpha'],'metrics':metrics_acc},
            'training_seconds':prior_seconds+time.perf_counter()-started,'checkpoint_sha256':{'macro':sha(out/'best_macro.pt'),'acc':sha(out/'best_acc.pt')},
            'plan_sha256':sha(PLAN),'data_manifest_sha256':sha(DATA/'manifest.json'),'confirmation_accessed':False}
    write(result_path,result);write(out/'progress.json',{'status':'complete','best_macro':result['best_macro'],'best_acc':result['best_acc']})
    return result

def check(job,plan,device):
    torch.manual_seed(plan['seed']);model=make_model(job['spec']).to(device);opt=optimizer(model,job,device)
    x=torch.randn(8,359,device=device);y=torch.arange(8,device=device)%10
    weights=torch.ones(10,device=device);loss=eager_step(model,opt,x,y,weights,job)
    assert math.isfinite(float(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    return {'name':job['name'],'counts':model.counts(),'loss':float(loss),'gradient_stats':gradient_stats(model)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['check','train']);ap.add_argument('--name');ap.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=ap.parse_args();torch.set_num_threads(4 if args.device=='cuda' else 16);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    if device.type=='cuda':assert torch.cuda.is_available() and 'L20' in torch.cuda.get_device_name(0)
    plan=json.loads(PLAN.read_text());validate_plan(plan)
    jobs=plan['jobs'] if args.name is None else [next(j for j in plan['jobs'] if j['name']==args.name)]
    if args.command=='check':
        rows=[check(j,plan,device) for j in jobs if (j['device']==args.device and not j.get('init_from'))]
        write(RUN/f'check_{args.device}.json',{'status':'passed','rows':rows,'plan_sha256':sha(PLAN),'confirmation_accessed':False})
    else:
        assert len(jobs)==1 and jobs[0]['device']==args.device
        try:train(jobs[0],plan,device)
        except Exception as e:
            write(RUN/'models'/jobs[0]['name']/'failure.json',{'status':'failed','error':repr(e),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
