"""Expanded-validation, paired-seed controls for the q9d12 architecture.

The fixed seed42 M8 pretraining is shared. Seeds vary newly added weights and
fine-tuning sample order; they are conditional fine-tuning replications, not
independent re-pretraining of M8. Pure dense MLPs receive 80 training epochs.
All original sources, folds and checkpoints remain immutable.
"""
import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import train_priority_14235 as trainer
import audit_priority_14235_results as auditor
import p1_matched_classical_comparison as capture
from global_hybrid_transformer_models import GlobalCoherentReadout, apply_gate, ry, rz
from priority_14235_models import make_model as priority_factory, _encode
from safe_capturable_adamw import SafeCapturableAdamW

ROOT = Path(__file__).resolve().parent
RUN = Path('/mnt/hdd/yxh/qml_paper_evidence_20261014/neural')
DATA = RUN / 'data'
PLAN = RUN / 'plan.json'
OLD_DATA = ROOT / 'hierarchical_lightgbm_tree/data'
EXP = ROOT / 'expanded_evaluation_20261014'
GPU = 'GPU-659579c9-052a-ec0f-637c-de675325d6d8'
BASE_OPTIMIZER, BASE_ORDER, BASE_GRADIENTS = trainer.optimizer, trainer.source_order, trainer.gradient_stats
ORIGINAL_ALPHA, ORIGINAL_LOAD_SPLIT = trainer.alpha_grid, trainer.load_split
sha, write = trainer.sha, trainer.write


def now(): return datetime.now(timezone.utc).isoformat()


class PureDenseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.variant = 'pure_mlp'
        self.network = nn.Sequential(nn.Linear(359,73), nn.LayerNorm(73), nn.GELU(),
                                     nn.Linear(73,207), nn.LayerNorm(207), nn.GELU(), nn.Linear(207,10))

    def forward(self, x, return_aux=False):
        logits = self.network(torch.atan(x))
        return (logits, {'base':logits}) if return_aux else logits

    def counts(self):
        total = sum(p.numel() for p in self.parameters())
        assert total == 44238
        return {'total':total,'quantum':0,'quantum_encoding':0,'classical':total,
                'pretrained_backbone':0,'variant':'pure_mlp','full_input_features':359,
                'hidden_widths':[73,207],'layernorm_layers':2,'fixed_random_projection':False}


class NoCNOTReadout(GlobalCoherentReadout):
    """Remove circuit CNOT gates; amplitude-encoded input may still be entangled."""
    def forward(self, amplitude, angles):
        state = amplitude.to(torch.complex64)
        for layer in range(self.depth):
            for q in range(self.qubits):
                w = self.theta[layer,q]
                state = apply_gate(state, rz(w[2]) @ ry(w[1]) @ rz(w[0]), q, self.qubits)
            phase = angles*self.frequency[layer] + self.shift[layer]
            state = state * torch.exp(-.5j*(phase @ self.zsign.T))
        probability = state.real.square() + state.imag.square()
        z = probability @ self.zsign
        xs, ys = [], []
        for q in range(self.qubits):
            overlap = state.conj()*state[:,getattr(self,'xflip_'+str(q))]
            xs.append(overlap.real.sum(-1))
            ys.append((overlap.imag*self.zsign[:,q]).sum(-1))
        return torch.cat((torch.stack(xs,-1),torch.stack(ys,-1),z),-1)


class QuadraticReadout(nn.Module):
    """540 meaningful weights over fixed bilinear features of all 359 inputs."""
    def __init__(self):
        super().__init__()
        weight = .1*torch.randn(27,20)
        self.fast_weight = nn.Parameter(weight[:,:12].clone())
        self.slow_weight = nn.Parameter(weight[:,12:].clone())
        generator = torch.Generator().manual_seed(20261014)
        left = torch.randint(0,359,(27,20,32),generator=generator)
        right = torch.randint(0,359,(27,20,32),generator=generator)
        sign = 2*torch.randint(0,2,(27,20,32),generator=generator).float()-1
        assert len(torch.unique(torch.cat((left.reshape(-1),right.reshape(-1))))) == 359
        self.register_buffer('left',left); self.register_buffer('right',right); self.register_buffer('sign',sign)

    def fast_parameters(self): return [self.fast_weight]
    def slow_parameters(self): return [self.slow_weight]

    def forward(self,x):
        raw = torch.atan(x)
        norm = torch.linalg.vector_norm(raw,dim=-1,keepdim=True)
        fallback = torch.zeros_like(raw); fallback[:,0] = 1.
        r = math.sqrt(359.)*torch.where(norm>1e-8,raw/norm.clamp_min(1e-8),fallback)
        features = (r[:,self.left]*r[:,self.right]*self.sign).sum(-1)/math.sqrt(32.)
        weight = torch.cat((self.fast_weight,self.slow_weight),-1)
        return torch.tanh((features*weight).sum(-1)/math.sqrt(20.))


def make_model(spec):
    kind, seed = spec['kind'], spec['seed']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if kind == 'pure_mlp': return PureDenseMLP()
        is_quantum = kind in ('quantum','no_cnot','frozen_quantum')
        model = priority_factory({'variant':'adamw_branchfirst' if is_quantum else 'classical_residual_control',
                                  'qubits':9,'depth':12,'registers':1,'fusion_hidden':128})
        if kind == 'no_cnot':
            state = model.readout.state_dict()
            model.readout = NoCNOTReadout(9,12)
            model.readout.load_state_dict(state)
        elif kind == 'matched_mlp':
            model.readout = capture.ProjectedMLPReadout()
            torch.manual_seed(seed)
            model.readout.first.reset_parameters()
            head = nn.Linear(10,27,bias=False)
            with torch.no_grad():
                model.readout.fast_head.copy_(head.weight[:,:2])
                model.readout.slow_head.copy_(head.weight[:,2:])
        elif kind == 'strong_mlp':
            torch.manual_seed(seed)
            model.readout = nn.Sequential(nn.Linear(359,32),nn.GELU(),nn.Linear(32,27))
        elif kind == 'quadratic':
            torch.manual_seed(seed)
            model.readout = QuadraticReadout()
        elif not is_quantum: raise ValueError(kind)
        # Identical shared head initialization for every branch at a given seed.
        torch.manual_seed(10000+seed)
        model.fusion_head = nn.Sequential(nn.Linear(147,128),nn.GELU(),nn.Linear(128,10))
        nn.init.zeros_(model.fusion_head[-1].weight); nn.init.zeros_(model.fusion_head[-1].bias)
        model.evidence_kind = kind
        model.evidence_seed = seed
        expected = 56109 if kind == 'strong_mlp' else 44238
        assert model.counts()['total'] == expected
        return model


def optimizer(model,job,device):
    if job['spec']['kind'] in ('matched_mlp','quadratic'):
        opt = capture.optimizer(model,job,device)
    else:
        opt = BASE_OPTIMIZER(model,job,device)
    if job['spec']['kind'] == 'pure_mlp':
        assert len(opt.param_groups) == 1 and opt.param_groups[0]['group_name'] == 'branch'
        group = opt.param_groups[0]
        group.update(start=job['learning_rate'],end=job['learning_rate']*.1)
        group['lr'].fill_(job['learning_rate'])
    if job['spec']['kind'] == 'frozen_quantum':
        for group in opt.param_groups:
            if group['group_name'] in ('theta','encoding'):
                group.update(start=0.,end=0.); group['lr'].zero_()
    return opt


def configure():
    trainer.RUN,trainer.PLAN,trainer.DATA = RUN,PLAN,DATA
    trainer.make_model,trainer.optimizer = make_model,optimizer
    auditor.RUN,auditor.PLAN,auditor.DATA,auditor.make_model = RUN,PLAN,DATA,make_model
    capture.RUN,capture.PLAN,capture.make_model = RUN,PLAN,make_model


def prepare_data():
    DATA.mkdir(parents=True,exist_ok=True)
    original = json.loads((OLD_DATA/'manifest.json').read_text())
    pop = np.load(original['data_sources']['population'])
    old_ids = np.load(OLD_DATA/'val_ids.npy')
    rest = np.setdiff1d(np.flatnonzero(pop['part']==1),old_ids)
    ids = np.concatenate((old_ids,rest))
    assert len(ids) == len(np.unique(ids)) == 105297
    fit = np.load(OLD_DATA/'fit_index.npy')
    assert not np.intersect1d(ids,fit).size
    assert not np.intersect1d(ids,np.load(ROOT/'data/confirmation2_ids.npz')['ids']).size
    bank = np.load(original['data_sources']['bank_features'],mmap_mode='r')
    normalization = np.load(ROOT/'data/features359.npz')
    raw = np.array(bank[ids],dtype=np.float64)
    med,scale = normalization['median'],normalization['scale']
    x = np.clip((np.where(np.isfinite(raw),raw,med)-med)/scale,-10,10).astype(np.float32)
    y = pop['y'][ids].astype(np.int64)
    assert np.array_equal(x[:10000],np.load(OLD_DATA/'val_x.npy'))
    assert np.array_equal(y[:10000],np.load(OLD_DATA/'val_y.npy'))
    for name,array in (('val_x.npy',x),('val_y.npy',y),('val_ids.npy',ids)):
        np.save(DATA/name,array)
    for name in original['outputs']:
        if name.startswith('val_'): continue
        target = DATA/name
        if target.exists(): assert target.resolve() == (OLD_DATA/name).resolve()
        else: target.symlink_to(OLD_DATA/name)
    manifest = copy.deepcopy(original)
    manifest.update(validation='Expanded source partition1:105297 objects, including the original10000 and95297 additional objects; selection only.',
                    fixed_split_sizes={'val':105297,'report':5000,'extra':20000},
                    expanded_val_class_counts=np.bincount(y,minlength=10).tolist(),
                    original_manifest_sha256=sha(OLD_DATA/'manifest.json'),
                    normalization_sha256=sha(ROOT/'data/features359.npz'),confirmation_accessed=False)
    manifest['outputs'] = {name:sha(DATA/name) for name in original['outputs']}
    write(DATA/'manifest.json',manifest)
    return manifest


def freeze():
    RUN.mkdir(parents=True,exist_ok=True)
    if PLAN.exists():
        plan=json.loads(PLAN.read_text());trainer.validate_plan(plan);return plan
    manifest=prepare_data()
    oldplan=json.loads((ROOT/'priority_14235_20261013/plan.json').read_text())
    jobs=[]
    def add(kind,seed,lr=None):
        name=kind+'_s'+str(seed)+(('_lr'+str(lr).replace('.','p')) if lr else '')
        jobs.append({'name':name,'priority_direction':1,'device':'cuda','epochs':80 if kind=='pure_mlp' else 40,
                     'objective':'standard','backbone_warmup_epochs':0 if kind=='pure_mlp' else 10,
                     'learning_rate':lr,'spec':{'kind':kind,'seed':seed}})
    # Fast direct-input baselines start first; two LR values are predeclared.
    add('pure_mlp',42,.001);add('pure_mlp',42,.003)
    for kind in ('quantum','matched_mlp','no_cnot','frozen_quantum','strong_mlp','quadratic'):add(kind,42)
    for seed in (43,44):
        for kind in ('quantum','matched_mlp'):add(kind,seed)
        add('pure_mlp',seed,.001);add('pure_mlp',seed,.003)
    assert len(jobs)==16
    plan=copy.deepcopy(oldplan)
    plan.update(jobs=jobs,main_result_models=[j['name'] for j in jobs],seed=42,
                counts={j['name']:make_model(j['spec']).counts() for j in jobs},
                data_manifest_sha256=sha(DATA/'manifest.json'),
                reference_plan_sha256=sha(ROOT/'priority_14235_20261013/plan.json'),
                pretrained_M8_sha256=sha(ROOT/'top10_newdata/models/mlp_rank08_window359_mlp_q6_ln1_wd0/best_macro.pt'),
                inference_scope='Report evaluation on the fixed 20,000-object test cohort; checkpoint/alpha selection uses validation only.',
                primary_endpoint='Macro-F1; ACC also independently selected and completely reported.',
                seed_replicates=[42,43,44],
                replication_scope='New branch initialization and fine-tuning object order vary by seed; shared M8 pretraining remains seed42. Pure dense models start fresh and train80epochs. Do not claim independent M8 re-pretraining.',
                training_order_seed='2026101100+model_seed; paired hybrid models have identical epoch ordering.',
                pure_MLP_selection='For each seed and endpoint select between the two predeclared LRs using expanded validation only; report both raw runs.',
                mechanism_screen='Single-seed42 exploratory trained controls: no CNOT in variational circuit, frozen quantum readout, stronger learned dense359->32->27 classical branch, fixed bilinear features with540 learned weights. Amplitude input may remain entangled even with no CNOT.',
                optimization_scope='Quantum-frozen control uses zero theta/encoding learning rates and exactly unchanged readout weights; optimizer moments may accumulate. Stored parameter count remains44238, effectively updated parameters43698.',
                output_storage=str(RUN),authorization='Active user goal: gather superiority, pureMLP/RF, mechanism and paper-ready transparent evidence.',
                confirmation_accessed=False)
    plan['source_sha256'].update({'paper_evidence_neural.py':sha(__file__),
                                 'p1_matched_classical_comparison.py':sha(ROOT/'p1_matched_classical_comparison.py'),
                                 'audit_priority_14235_results.py':sha(ROOT/'audit_priority_14235_results.py')})
    write(PLAN,plan);trainer.validate_plan(plan)
    return plan


def preflight(plan):
    x=torch.from_numpy(np.load(DATA/'val_x.npy')[:64].copy())
    y=torch.from_numpy(np.load(DATA/'val_y.npy')[:64].copy())
    rows=[]
    representatives=[next(j for j in plan['jobs'] if j['spec']['kind']==kind) for kind in
                     ('pure_mlp','quantum','matched_mlp','no_cnot','frozen_quantum','strong_mlp','quadratic')]
    shared=None
    for job in representatives:
        model=make_model(job['spec']).eval()
        if job['spec']['kind']!='pure_mlp':
            shared_prefixes=('backbone.','classical_norm.','representation_norm.','fusion_head.')
            common={k:v.clone() for k,v in model.state_dict().items() if k.startswith(shared_prefixes)}
            if shared is None:shared=common
            else:assert common.keys()==shared.keys() and all(torch.equal(v,shared[k]) for k,v in common.items())
        eager=make_model(job['spec']).cuda().train();graphed=make_model(job['spec']).cuda().train()
        initial={k:v.detach().clone() for k,v in eager.state_dict().items()}
        device=torch.device('cuda');xe,ye=x.cuda(),y.cuda();weights=torch.ones(10,device=device)
        oe,og=optimizer(eager,job,device),optimizer(graphed,job,device)
        graph=trainer.GraphStep(graphed,og,xe,ye,weights,job)
        for _ in range(3):
            loss=trainer.eager_step(eager,oe,xe,ye,weights,job);other=graph(xe,ye)
            assert math.isfinite(float(loss)) and math.isfinite(float(other))
        torch.cuda.synchronize()
        error=max(float((v-graphed.state_dict()[k]).abs().max()) for k,v in eager.state_dict().items() if v.is_floating_point())
        assert error<3e-5,(job['name'],error)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in eager.parameters())
        if job['spec']['kind']!='pure_mlp':
            assert all(torch.equal(initial[k],eager.state_dict()[k]) for k in initial if k.startswith('backbone.'))
        if job['spec']['kind']=='frozen_quantum':
            assert all(torch.equal(initial[k],eager.state_dict()[k]) for k in initial if k.startswith('readout.'))
        rows.append({'name':job['name'],'counts':model.counts(),'eager_graph_max_error':error,'finite_gradients':True,
                     'group_parameters':{g['group_name']:sum(p.numel() for p in g['params']) for g in oe.param_groups}})
        del eager,graphed,oe,og,graph;torch.cuda.empty_cache()
    # Physical reference for both circuit topologies on one input, using the
    # correct least-significant-wire amplitude ordering.
    import pennylane as qml
    physical=[]
    for kind in ('quantum','no_cnot'):
        model=make_model({'kind':kind,'seed':42}).eval()
        amp,angles=_encode(x[:1],9,model.feature_permutations[0],model.zero_basis)
        amplitude=amp[0].numpy().astype(float);amplitude/=np.linalg.norm(amplitude)
        angle=angles[0].numpy().astype(float);readout=model.readout
        theta=readout.theta.detach().numpy().astype(float)
        frequency=readout.frequency.detach().numpy().astype(float);shift=readout.shift.detach().numpy().astype(float)
        device=qml.device('default.qubit',wires=9)
        @qml.qnode(device,diff_method='adjoint')
        def circuit(weights):
            qml.AmplitudeEmbedding(amplitude,wires=list(reversed(range(9))),normalize=True)
            for layer in range(12):
                for q in range(9):
                    qml.RZ(weights[layer,q,0],wires=q);qml.RY(weights[layer,q,1],wires=q);qml.RZ(weights[layer,q,2],wires=q)
                if kind=='quantum':
                    for q in range(9):qml.CNOT(wires=[q,(q+1)%9])
                for q in range(9):qml.RZ(angle[q]*frequency[layer,q]+shift[layer,q],wires=q)
            return tuple(qml.expval(op(q)) for op in (qml.PauliX,qml.PauliY,qml.PauliZ) for q in range(9))
        with torch.no_grad():actual=readout(amp,angles)[0].numpy()
        reference=np.asarray(circuit(theta))
        error=float(np.max(np.abs(actual-reference)));assert error<5e-6,(kind,error)
        physical.append({'kind':kind,'PennyLane_float64_max_error':error,'cnot_removed_does_not_imply_separable_input':True})
    result={'status':'passed','plan_sha256':sha(PLAN),'models':rows,'physical_reference':physical,
            'shared_initial_tensors_equal_within_seed':True,'confirmation_accessed':False}
    write(RUN/'preflight.json',result);return result


def audit_protocol(job):
    folder=RUN/'models'/job['name'];audit=json.loads((folder/'result_audit.json').read_text())
    initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=True)
    latest=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
    assert latest['epoch']==job['epochs'] and latest['cursor']==0
    assert [r['epoch'] for r in latest['history']]==list(range(job['epochs']+1))
    if job['spec']['kind']!='pure_mlp':
        epoch10=torch.load(folder/'epoch_checkpoints/epoch_010.pt',map_location='cpu',weights_only=True)
        assert all(torch.equal(initial[k],epoch10[k]) for k in initial if k.startswith('backbone.'))
        for row in latest['history'][1:]:assert (row['learning_rates']['backbone']==0)==(row['epoch']<=10)
    frozen=job['spec']['kind']=='frozen_quantum'
    if frozen:assert all(torch.equal(initial[k],latest['model'][k]) for k in initial if k.startswith('readout.'))
    expected_steps=job['epochs']*2473
    assert audit['optimizer_steps']==expected_steps
    result={'status':'passed','name':job['name'],'epochs':job['epochs'],'optimizer_steps':expected_steps,
            'complete_epoch_weights':len(list((folder/'epoch_checkpoints').glob('epoch_*.pt'))),
            'readout_frozen_and_verified':frozen,'data_manifest_sha256':sha(DATA/'manifest.json'),
            'result_audit_sha256':sha(folder/'result_audit.json'),'replication_scope':'conditional on fixed pretrained M8',
            'confirmation_accessed':False}
    assert result['complete_epoch_weights']==job['epochs']+1
    write(folder/'protocol_audit.json',result)


def train_one(job,plan):
    trainer.source_order=lambda y,epoch,batch,unused_seed,k:BASE_ORDER(y,epoch,batch,2026101100+job['spec']['seed'],k)
    capture.capture_validation_checkpoints(job,plan)
    trainer.train(job,plan,torch.device('cuda'))
    trainer.make_model,trainer.alpha_grid,trainer.load_split=make_model,ORIGINAL_ALPHA,ORIGINAL_LOAD_SPLIT
    torch.cuda.empty_cache()
    auditor.audit(job['name'])
    audit_protocol(job)


def scheduler(plan):
    check=json.loads((RUN/'preflight.json').read_text());assert check['status']=='passed' and check['plan_sha256']==sha(PLAN)
    state={'status':'running','pid':os.getpid(),'started_utc':now(),'plan_sha256':sha(PLAN),'workers':[],'completed':[],'failed':{}}
    active={};attempted=set()
    while True:
        for name,(process,job) in list(active.items()):
            code=process.poll()
            if code is None:continue
            del active[name]
            folder=RUN/'models'/name
            good=code==0 and (folder/'result_audit.json').exists() and (folder/'protocol_audit.json').exists()
            if good:state['completed'].append(name)
            else:state['failed'][name]={'exit_code':code,'error':json.loads((folder/'failure.json').read_text()) if (folder/'failure.json').exists() else 'Inspect worker log'}
        for job in plan['jobs']:
            if len(active)>=2:break
            name=job['name']
            if name in attempted:continue
            attempted.add(name);folder=RUN/'models'/name
            if (folder/'protocol_audit.json').exists():state['completed'].append(name);continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            (RUN/'logs').mkdir(exist_ok=True)
            with (RUN/'logs'/(name+'.log')).open('ab',buffering=0) as log:
                p=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'train','--name',name],cwd=ROOT,
                    env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[name]=(p,job)
            write(RUN/'workers'/(name+'.json'),{'pid':p.pid,'started_utc':now(),'command':'paper_evidence_neural.py train --name '+name})
        state.update(workers=[{'name':name,'pid':p.pid} for name,(p,j) in active.items()],checked_utc=now())
        if not active and len(attempted)==len(plan['jobs']):
            state['status']='complete' if not state['failed'] else 'finished_with_failures'
        write(RUN/'state.json',state)
        if state['status']!='running':return state
        time.sleep(5)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','check','launch','schedule','train','status'))
    parser.add_argument('--name');args=parser.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    configure();plan=freeze()
    if args.command=='freeze':out={'status':'frozen','plan_sha256':sha(PLAN),'jobs':len(plan['jobs']),'validation':json.loads((DATA/'manifest.json').read_text())['expanded_val_class_counts']}
    elif args.command=='check':out=preflight(plan)
    elif args.command=='schedule':out=scheduler(plan)
    elif args.command=='train':
        job=next(j for j in plan['jobs'] if j['name']==args.name)
        try:train_one(job,plan)
        except Exception as error:
            write(RUN/'models'/job['name']/'failure.json',{'error':repr(error),'traceback':traceback.format_exc()});raise
        out={'status':'complete','name':job['name']}
    elif args.command=='status':
        state=json.loads((RUN/'state.json').read_text()) if (RUN/'state.json').exists() else {}
        models=[]
        for job in plan['jobs']:
            folder=RUN/'models'/job['name']
            progress=json.loads((folder/'progress.json').read_text()) if (folder/'progress.json').exists() else None
            audit=json.loads((folder/'result_audit.json').read_text()) if (folder/'result_audit.json').exists() else None
            models.append({'name':job['name'],'progress':progress,'audit':audit,'protocol_passed':(folder/'protocol_audit.json').exists()})
        live=[]
        for worker in state.get('workers',[]):
            path=Path('/proc')/str(worker['pid'])/'cmdline'
            live.append({**worker,'alive':path.exists() and 'paper_evidence_neural.py' in path.read_bytes().decode()})
        out={'state':state,'live_workers':live,'models':models,'plan_sha256':sha(PLAN)}
    else:
        receipt=json.loads((RUN/'launch.json').read_text()) if (RUN/'launch.json').exists() else {}
        proc=Path('/proc')/str(receipt.get('pid'))/'cmdline'
        if proc.exists() and 'paper_evidence_neural.py' in proc.read_bytes().decode():out=dict(receipt,already_running=True)
        else:
            assert json.loads((RUN/'preflight.json').read_text())['status']=='passed'
            with (RUN/'scheduler.log').open('ab',buffering=0) as log:
                p=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'schedule'],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
            out={'status':'launched','pid':p.pid,'utc':now(),'plan_sha256':sha(PLAN),'max_concurrent_gpu_workers':2};write(RUN/'launch.json',out)
    print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
