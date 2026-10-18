"""Additional fixed-budget full-input interaction controls, after the six-job supplement."""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import paper_evidence_neural as neural
from train_priority_14235 import sha,write

ORIGINAL=Path('/mnt/hdd/yxh/qml_paper_evidence_20261014/neural')
PREVIOUS=ORIGINAL.parent/'followup_seed44_seed45'
RUN=ORIGINAL.parent/'fullinteraction_seed44_seed45'
PLAN=RUN/'plan.json'
DATA=ORIGINAL/'data'
GPU='GPU-659579c9-052a-ec0f-637c-de675325d6d8'
ORIGINAL_SHA='7a94001fe61846f105a460cedb3804673f26a7e8b1bb4ceab13cb13c38e72ee7'
PREVIOUS_SHA='8fce3b618bb8ed92015825019aa3f21902c66f52f658c5ca50b3e8ef5604ce48'


def configure():
    neural.RUN,neural.PLAN,neural.DATA=RUN,PLAN,DATA
    neural.configure()


def freeze():
    assert sha(ORIGINAL/'plan.json')==ORIGINAL_SHA and sha(PREVIOUS/'plan.json')==PREVIOUS_SHA
    RUN.mkdir(parents=True,exist_ok=True)
    if PLAN.exists():
        plan=json.loads(PLAN.read_text())
        neural.trainer.validate_plan(plan)
        return plan
    plan=copy.deepcopy(json.loads((ORIGINAL/'plan.json').read_text()))
    template=next(j for j in plan['jobs'] if j['spec']['kind']=='quadratic')
    jobs=[]
    for seed in (44,45):
        job=copy.deepcopy(template)
        job['name']='quadratic_s'+str(seed)
        job['spec']['seed']=seed
        jobs.append(job)
    plan.update(jobs=jobs,main_result_models=[j['name'] for j in jobs],
                counts={j['name']:neural.make_model(j['spec']).counts() for j in jobs},
                original_neural_plan_sha256=ORIGINAL_SHA,preceding_supplement_plan_sha256=PREVIOUS_SHA,
                primary_comparisons=['quantum_s44 vs quadratic_s44','quantum_s45 vs quadratic_s45'],
                design='Retain all359 inputs for fixed bilinear features with540 learned weights; no fixed26-dimensional linear input projection. Replicate the original seed42 quadratic control at predeclared seeds44 and45.',
                rationale='Synthetic projection example identified an alternative explanation for the matched-MLP branch comparison. This stronger rival explanation must be tested, not treated as quantum-specific benefit.',
                inference_scope='Report evaluation on the fixed 20,000-object test cohort; checkpoint/alpha selection uses validation only.',
                training_protocol='Same M8, paired shared tensors, optimizer groups, class weights, 40epochs, and same-seed source ordering as quantum reference.',
                interaction_scope='Fixed randomly sampled normalized pair products cover all359 inputs; does not assert that every possible interaction is present or that the mapping is information preserving.',
                launch_gate='Wait for preceding six-job supplement to be terminal and all same-user GPU2 contexts to clear before GPU preflight or launch.',
                output_storage=str(RUN),confirmation_accessed=False)
    plan['source_sha256']['paper_evidence_fullinteraction.py']=sha(__file__)
    write(PLAN,plan)
    neural.trainer.validate_plan(plan)
    return plan


def ready():
    state_path=PREVIOUS/'state.json'
    if not state_path.exists():return False
    state=json.loads(state_path.read_text())
    if state['status'] not in ('complete','finished_with_failures') or state.get('workers'):return False
    result=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    for line in result.stdout.splitlines():
        uuid,pid=[value.strip() for value in line.split(',')]
        if uuid!=GPU:continue
        proc=Path('/proc')/pid
        try:
            if proc.stat().st_uid==os.getuid():return False
        except FileNotFoundError:pass
    return True


def check(plan):
    assert ready(),'GPU checks deferred until preceding study is terminal'
    x=torch.from_numpy(np.load(DATA/'fit_x.npy',mmap_mode='r')[:64].copy()).cuda()
    y=torch.from_numpy(np.load(DATA/'fit_y.npy',mmap_mode='r')[:64].copy()).cuda()
    count=np.bincount(np.load(DATA/'fit_y.npy'),minlength=10)
    weights=torch.tensor(count.sum()/(10*count),dtype=torch.float32,device='cuda')
    rows=[]
    for job in plan['jobs']:
        reference=neural.make_model({'kind':'quantum','seed':job['spec']['seed']})
        eager=neural.make_model(job['spec']).cuda().train()
        graphed=neural.make_model(job['spec']).cuda().train()
        initial={key:value.detach().cpu().clone() for key,value in eager.state_dict().items()}
        assert all(torch.equal(value,reference.state_dict()[key]) for key,value in initial.items() if not key.startswith('readout.'))
        assert sum(p.numel() for p in eager.readout.parameters())==540
        assert len(torch.unique(torch.cat((eager.readout.left.flatten(),eager.readout.right.flatten()))))==359
        oe=neural.optimizer(eager,job,torch.device('cuda'))
        og=neural.optimizer(graphed,job,torch.device('cuda'))
        graph=neural.trainer.GraphStep(graphed,og,x,y,weights,job)
        for _ in range(3):
            a=neural.trainer.eager_step(eager,oe,x,y,weights,job)
            b=graph(x,y)
            assert bool(torch.isfinite(a)) and bool(torch.isfinite(b))
        error=max(float((value-graphed.state_dict()[key]).abs().max()) for key,value in eager.state_dict().items() if value.is_floating_point())
        assert error<3e-5
        assert all(torch.equal(initial[key],eager.state_dict()[key].cpu()) for key in initial if key.startswith('backbone.'))
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in eager.parameters())
        rows.append({'name':job['name'],'counts':eager.counts(),'eager_graph_error':error,'shared_initial_tensors_equal':True,'all359_input_coverage':True})
        del reference,eager,graphed,oe,og,graph
        torch.cuda.empty_cache()
    out={'status':'passed','plan_sha256':sha(PLAN),'rows':rows,'confirmation_accessed':False}
    write(RUN/'preflight.json',out)
    return out


def schedule(plan):
    state={'status':'waiting_for_preceding_study','pid':os.getpid(),'plan_sha256':sha(PLAN),'workers':[],'completed':[],'failed':{}}
    while not ready():
        write(RUN/'state.json',state)
        time.sleep(60)
    check(plan)
    active={}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    (RUN/'logs').mkdir(exist_ok=True)
    for job in plan['jobs']:
        name=job['name']
        if (RUN/'models'/name/'protocol_audit.json').exists():
            state['completed'].append(name)
            continue
        with (RUN/'logs'/(name+'.log')).open('ab',buffering=0) as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'train','--name',name],cwd=ROOT,
                env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[name]=child
    while active:
        state.update(status='running',workers=[{'name':name,'pid':child.pid} for name,child in active.items()])
        write(RUN/'state.json',state)
        for name,child in list(active.items()):
            code=child.poll()
            if code is None:continue
            del active[name]
            if code==0 and (RUN/'models'/name/'protocol_audit.json').exists():state['completed'].append(name)
            else:state['failed'][name]={'exit_code':code,'error':json.loads((RUN/'models'/name/'failure.json').read_text()) if (RUN/'models'/name/'failure.json').exists() else 'Inspect log'}
        time.sleep(5)
    state.update(status='complete' if not state['failed'] else 'finished_with_failures',workers=[])
    write(RUN/'state.json',state)
    return state


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('freeze','check','launch','schedule','train','status'))
    parser.add_argument('--name')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    configure()
    plan=freeze()
    if args.command=='freeze':out={'status':'frozen','plan_sha256':sha(PLAN),'jobs':[j['name'] for j in plan['jobs']]}
    elif args.command=='check':out=check(plan)
    elif args.command=='schedule':out=schedule(plan)
    elif args.command=='train':
        job=next(j for j in plan['jobs'] if j['name']==args.name)
        try:neural.train_one(job,plan)
        except Exception:
            write(RUN/'models'/job['name']/'failure.json',{'error':traceback.format_exc()})
            raise
        out={'status':'complete','name':job['name']}
    elif args.command=='status':
        state=json.loads((RUN/'state.json').read_text()) if (RUN/'state.json').exists() else {}
        workers=[]
        for worker in state.get('workers',[]):
            proc=Path('/proc')/str(worker['pid'])/'cmdline'
            workers.append({**worker,'alive':proc.exists() and 'paper_evidence_fullinteraction.py' in proc.read_bytes().decode()})
        out={'state':state,'live_workers':workers,'plan_sha256':sha(PLAN)}
    else:
        receipt=json.loads((RUN/'launch.json').read_text()) if (RUN/'launch.json').exists() else {}
        prior_state=json.loads((RUN/'state.json').read_text()) if (RUN/'state.json').exists() else {}
        proc=Path('/proc')/str(receipt.get('pid'))/'cmdline'
        living=[]
        for worker in prior_state.get('workers',[]):
            handle=Path('/proc')/str(worker['pid'])/'cmdline'
            if handle.exists() and 'paper_evidence_fullinteraction.py' in handle.read_bytes().decode():living.append(worker)
        if prior_state.get('status') in ('complete','finished_with_failures'):
            out={'status':'already_finished','plan_sha256':sha(PLAN),'state':prior_state}
        elif proc.exists() and 'paper_evidence_fullinteraction.py' in proc.read_bytes().decode():out=dict(receipt,already_running=True)
        elif living:
            out={'status':'orphan_workers_detected_no_restart','live_workers':living,'plan_sha256':sha(PLAN)}
        else:
            with (RUN/'scheduler.log').open('ab',buffering=0) as log:
                child=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'schedule'],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
            out={'status':'launched_or_deferred','pid':child.pid,'plan_sha256':sha(PLAN),'max_concurrent_gpu_workers':2}
            write(RUN/'launch.json',out)
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()
