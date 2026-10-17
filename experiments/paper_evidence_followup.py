"""Immutable supplement: seed44 mechanisms and a predeclared fourth seed45."""
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import paper_evidence_neural as neural
from train_priority_14235 import sha, write

ORIGINAL_RUN = Path('/mnt/hdd/yxh/qml_paper_evidence_20261014/neural')
RUN = ORIGINAL_RUN.parent / 'followup_seed44_seed45'
PLAN = RUN / 'plan.json'
DATA = ORIGINAL_RUN / 'data'
GPU = 'GPU-659579c9-052a-ec0f-637c-de675325d6d8'
ORIGINAL_PLAN_SHA = '7a94001fe61846f105a460cedb3804673f26a7e8b1bb4ceab13cb13c38e72ee7'


def now():
    return datetime.now(timezone.utc).isoformat()


def configure():
    neural.RUN, neural.PLAN, neural.DATA = RUN, PLAN, DATA
    neural.configure()


def freeze():
    assert sha(ORIGINAL_RUN / 'plan.json') == ORIGINAL_PLAN_SHA
    RUN.mkdir(parents=True, exist_ok=True)
    if PLAN.exists():
        plan = json.loads(PLAN.read_text())
        neural.trainer.validate_plan(plan)
        return plan
    plan = copy.deepcopy(json.loads((ORIGINAL_RUN / 'plan.json').read_text()))
    templates = {j['spec']['kind']: j for j in plan['jobs'] if j['spec']['seed'] == 42}
    jobs = []
    # These independent mechanisms come first; both retain exactly the same
    # seed44 initial shared head/backbone tensors and training-object ordering.
    for kind, seed, lr in [('frozen_quantum',44,None), ('no_cnot',44,None),
                           ('quantum',45,None), ('matched_mlp',45,None),
                           ('pure_mlp',45,.001), ('pure_mlp',45,.003)]:
        job = copy.deepcopy(templates[kind])
        job['spec']['seed'] = seed
        job['learning_rate'] = lr
        job['name'] = kind+'_s'+str(seed)+(('_lr'+str(lr).replace('.','p')) if lr else '')
        jobs.append(job)
    plan.update(jobs=jobs, main_result_models=[j['name'] for j in jobs],
                counts={j['name']: neural.make_model(j['spec']).counts() for j in jobs},
                original_neural_plan_sha256=ORIGINAL_PLAN_SHA, created_utc=now(),
                output_storage=str(RUN), seed_replicates=[42,43,44,45],
                supplement_design='Two seed44 mechanism controls and an additional predeclared seed45 quantum/matched/pureMLP pair. Existing seed43 original audit failure remains excluded and reported.',
                seed45_rationale='A further fixed seed after a numerical audit failure; not selected from development performance. Report all attempted seeds including failed43.',
                primary_comparisons=['quantum_s44 vs frozen_quantum_s44','quantum_s44 vs no_cnot_s44',
                                     'quantum_s45 vs matched_mlp_s45','quantum_s45 vs validation-selected pure_mlp_s45'],
                selection='Original strict CPU/GPU grid and prediction audit retained. Each checkpoint/alpha and pureMLP learning rate use validation105297 only.',
                unchanged='Same factories, data, 40/80 epochs, optimizer schedules, initialization, and ordering as original study.',
                confirmation_accessed=False)
    plan['source_sha256']['paper_evidence_followup.py'] = sha(__file__)
    write(PLAN, plan)
    neural.trainer.validate_plan(plan)
    return plan


def check(plan):
    x = torch.from_numpy(np.load(DATA / 'fit_x.npy', mmap_mode='r')[:64].copy())
    y = torch.from_numpy(np.load(DATA / 'fit_y.npy', mmap_mode='r')[:64].copy())
    rows = []
    representatives = list(plan['jobs'][:4]) + [plan['jobs'][4]]
    for job in representatives:
        model = neural.make_model(job['spec']).eval()
        initial = {k:v.clone() for k,v in model.state_dict().items()}
        kind = job['spec']['kind']
        if job['spec']['seed'] == 44:
            original = torch.load(ORIGINAL_RUN / 'models/quantum_s44/initial.pt', map_location='cpu', weights_only=True)
            assert initial.keys() == original.keys()
            assert all(torch.equal(v, original[k]) for k,v in initial.items())
        eager = neural.make_model(job['spec']).cuda().train()
        graph_model = neural.make_model(job['spec']).cuda().train()
        assert all(torch.equal(v,graph_model.state_dict()[k]) for k,v in eager.state_dict().items())
        oe = neural.optimizer(eager, job, torch.device('cuda'))
        og = neural.optimizer(graph_model, job, torch.device('cuda'))
        xe,ye = x.cuda(), y.cuda()
        freq = np.bincount(np.load(DATA / 'fit_y.npy'),minlength=10)
        weights = torch.tensor(freq.sum()/(10*freq),dtype=torch.float32,device='cuda')
        graph = neural.trainer.GraphStep(graph_model,og,xe,ye,weights,job)
        for _ in range(3):
            a = neural.trainer.eager_step(eager,oe,xe,ye,weights,job)
            b = graph(xe,ye)
            assert math.isfinite(float(a)) and math.isfinite(float(b))
        torch.cuda.synchronize()
        error = max(float((v-graph_model.state_dict()[k]).abs().max()) for k,v in eager.state_dict().items() if v.is_floating_point())
        assert error < 3e-5, (job['name'],error)
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in eager.parameters())
        if kind != 'pure_mlp':
            assert all(torch.equal(initial[k],eager.state_dict()[k].cpu()) for k in initial if k.startswith('backbone.'))
        if kind == 'frozen_quantum':
            assert all(torch.equal(initial[k],eager.state_dict()[k].cpu()) for k in initial if k.startswith('readout.'))
        if kind == 'quantum':
            assert float((eager.readout.theta.detach().cpu()-initial['readout.theta']).abs().max()) > 0
        rows.append({'name': job['name'], 'counts':model.counts(),'eager_graph_max_error':error,
                     'shared_seed44_initial_tensors_equal':job['spec']['seed']==44,'finite_gradients':True})
        del eager,graph_model,oe,og,graph
        torch.cuda.empty_cache()
    receipt = {'status':'passed','plan_sha256':sha(PLAN),'rows':rows,'confirmation_accessed':False}
    write(RUN / 'preflight.json',receipt)
    return receipt


def schedule(plan):
    assert json.loads((RUN/'preflight.json').read_text())['plan_sha256'] == sha(PLAN)
    state = {'status':'running','pid':os.getpid(),'started_utc':now(),'completed':[],'failed':{},'workers':[], 'plan_sha256':sha(PLAN)}
    active, attempted = {}, set()
    while True:
        for name,(proc,job) in list(active.items()):
            code = proc.poll()
            if code is None: continue
            del active[name]
            folder = RUN / 'models' / name
            if code == 0 and (folder/'protocol_audit.json').exists():
                state['completed'].append(name)
            else:
                state['failed'][name] = {'exit_code':code,'failure':json.loads((folder/'failure.json').read_text()) if (folder/'failure.json').exists() else 'Inspect log'}
        for job in plan['jobs']:
            if len(active) >= 2: break
            name = job['name']
            if name in attempted: continue
            attempted.add(name)
            if (RUN/'models'/name/'protocol_audit.json').exists():
                state['completed'].append(name)
                continue
            (RUN/'logs').mkdir(exist_ok=True)
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            with (RUN/'logs'/(name+'.log')).open('ab',buffering=0) as log:
                proc = subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'train','--name',name],
                    cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[name] = (proc,job)
        state.update(workers=[{'name':n,'pid':p.pid} for n,(p,j) in active.items()],checked_utc=now())
        if not active and len(attempted) == len(plan['jobs']):
            state['status'] = 'complete' if not state['failed'] else 'finished_with_failures'
        write(RUN/'state.json',state)
        if state['status'] != 'running': return state
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('freeze','check','launch','schedule','train','status'))
    parser.add_argument('--name')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    configure()
    plan = freeze()
    if args.command == 'freeze':
        out = {'status':'frozen','plan_sha256':sha(PLAN),'jobs':[j['name'] for j in plan['jobs']]}
    elif args.command == 'check': out = check(plan)
    elif args.command == 'schedule': out = schedule(plan)
    elif args.command == 'train':
        job = next(j for j in plan['jobs'] if j['name'] == args.name)
        try: neural.train_one(job,plan)
        except Exception:
            write(RUN/'models'/job['name']/'failure.json',{'error':traceback.format_exc()})
            raise
        out = {'status':'complete','name':job['name']}
    elif args.command == 'status':
        state = json.loads((RUN/'state.json').read_text()) if (RUN/'state.json').exists() else {}
        workers = []
        for worker in state.get('workers',[]):
            proc = Path('/proc')/str(worker['pid'])/'cmdline'
            workers.append({**worker,'alive':proc.exists() and 'paper_evidence_followup.py' in proc.read_bytes().decode()})
        out = {'state':state,'live_workers':workers,'plan_sha256':sha(PLAN)}
    else:
        assert json.loads((RUN/'preflight.json').read_text())['status'] == 'passed'
        path = RUN/'launch.json'
        receipt = json.loads(path.read_text()) if path.exists() else {}
        proc = Path('/proc')/str(receipt.get('pid'))/'cmdline'
        if proc.exists() and 'paper_evidence_followup.py' in proc.read_bytes().decode():
            out = dict(receipt,already_running=True)
        else:
            with (RUN/'scheduler.log').open('ab',buffering=0) as log:
                child = subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'schedule'],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
            out = {'status':'launched','pid':child.pid,'started_utc':now(),'plan_sha256':sha(PLAN),'max_concurrent_gpu_workers':2}
            write(path,out)
    print(json.dumps(out),flush=True)


if __name__ == '__main__': main()
