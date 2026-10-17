"""CPU-only fixed-checkpoint inference for two full-input interaction controls."""
import argparse
import hashlib
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
from train_priority_14235 import predict,sha,write
from evaluate_expanded_p1 import metrics
from paper_evidence_references import paired_bootstrap

BASE=neural.RUN.parent
TRAIN=BASE/'fullinteraction_seed44_seed45'
RUN=TRAIN/'evaluation'
EXP=ROOT/'expanded_evaluation_20261014'
ENDPOINTS=[('best_macro','best_macro'),('best_acc','best_accuracy')]


def read(path):return json.loads(Path(path).read_text())


def freeze(expected):
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'plan.json').exists():
        plan=read(RUN/'plan.json')
        assert sha(TRAIN/'plan.json')==plan['training_plan_sha256']
        assert sha(__file__)==plan['source_sha256']
        if expected:assert expected==plan['training_plan_sha256']
        return plan
    assert expected and sha(TRAIN/'plan.json')==expected
    training=read(TRAIN/'plan.json')
    assert [j['name'] for j in training['jobs']]==['quadratic_s44','quadratic_s45']
    assert training['original_neural_plan_sha256']=='7a94001fe61846f105a460cedb3804673f26a7e8b1bb4ceab13cb13c38e72ee7'
    plan={'status':'frozen_before_new_control_performance','training_plan_sha256':expected,
          'source_sha256':sha(__file__),'data_sha256':read(BASE/'development/plan.json')['data_sha256'],
          'reference_development_plan_sha256':sha(BASE/'development/plan.json'),
          'seed45_reference_evaluation_plan_sha256':sha(BASE/'followup_seed44_seed45/evaluation/plan.json'),
          'bootstrap_helper_sha256':sha(ROOT/'paper_evidence_references.py'),
          'selection':'Both validation-only endpoints. No selection on development outcomes.',
          'inference':'CPU FP32, four threads, batch256, two complete independent model reloads.',
          'comparisons':'Same-seed quantum minus full-input quadratic control at44/45, both selectors and both nested scopes.',
          'draws':10000,'expected_rows':4,'expected_comparisons':8,
          'interval_scope':'Pointwise exploratory paired-object95% percentile intervals; no search/seed/multiplicity uncertainty.',
          'missing_references':'Do not replace a failed quantum run with a favorable other seed. Disclose missing paired results.',
          'cohort_scope':'65631 exposed development objects;40631 nested historical subset, not independent replication.',
          'confirmation_accessed':False}
    write(RUN/'plan.json',plan)
    return plan


def evaluate(job,selector,prefix,x,y,ids,logweights,plan,state):
    key=job['name']+'_'+selector
    destination=RUN/(key+'_result.json')
    if destination.exists():return read(destination)
    folder=TRAIN/'models'/job['name']
    audit,protocol=read(folder/'result_audit.json'),read(folder/'protocol_audit.json')
    assert audit['status']==protocol['status']=='passed'
    assert audit['plan_sha256']==plan['training_plan_sha256']
    assert protocol['result_audit_sha256']==sha(folder/'result_audit.json')
    selected=audit[selector]
    checkpoint=folder/(prefix+'.pt')
    assert sha(checkpoint)==selected['checkpoint_sha256']
    raws=[]
    for index in (1,2):
        state.update(stage='CPU_inference',current=key,pass_number=index)
        write(RUN/'state.json',state)
        model=neural.make_model(job['spec']).eval()
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
        assert sum(p.numel() for p in model.readout.parameters())==540
        raws.append(predict(model,torch.from_numpy(x),batch=256))
    predictions=[(raw-selected['alpha']*logweights).argmax(1) for raw in raws]
    assert np.array_equal(predictions[0],predictions[1])
    pred=predictions[0]
    checks={}
    for split,start,end in [('report',0,5000),('extra',5000,25000)]:
        stored=np.load(folder/(split+'_'+prefix+'_predictions.npz'))
        assert np.array_equal(stored['ids'],ids[start:end]) and np.array_equal(stored['y'],y[start:end])
        assert np.array_equal(stored['pred'],pred[start:end])
        checks[split]={'objects':end-start,'prediction_disagreements':0}
    path=RUN/(key+'_predictions.npz')
    np.savez_compressed(path,ids=ids,y=y,pred=pred,logits=raws[0],alpha=np.asarray(selected['alpha']))
    row={'status':'passed','name':job['name'],'kind':'quadratic','seed':job['spec']['seed'],
         'selection':selector,'epoch':selected['epoch'],'alpha':selected['alpha'],'parameters':audit['counts']['total'],
         'validation':selected['metrics']['validation'],
         'metrics':{'development':metrics(y,pred),'previously_unused':metrics(y[25000:],pred[25000:])},
         'predictions_sha256':sha(path),'checkpoint_sha256':sha(checkpoint),'old_prediction_checks':checks,
         'complete_reload_prediction_disagreements':0,'plan_sha256':sha(RUN/'plan.json'),
         'training_audit_sha256':sha(folder/'result_audit.json'),'protocol_audit_sha256':sha(folder/'protocol_audit.json'),
         'confirmation_accessed':False}
    write(destination,row)
    return row


def compare(row,ids,y,plan):
    seed,selector=row['seed'],row['selection']
    folder=BASE/'development' if seed==44 else BASE/'followup_seed44_seed45/evaluation'
    reference_path=folder/('quantum_s'+str(seed)+'_'+selector+'_result.json')
    if not reference_path.exists():return None
    reference=read(reference_path)
    assert reference['status']=='passed' and reference['complete_reload_prediction_disagreements']==0
    expected=plan['reference_development_plan_sha256'] if seed==44 else plan['seed45_reference_evaluation_plan_sha256']
    assert reference['plan_sha256']==expected
    qpath=folder/('quantum_s'+str(seed)+'_'+selector+'_predictions.npz')
    cpath=RUN/(row['name']+'_'+selector+'_predictions.npz')
    assert sha(qpath)==reference['predictions_sha256'] and sha(cpath)==row['predictions_sha256']
    q,c=np.load(qpath),np.load(cpath)
    for data in (q,c):
        assert np.array_equal(data['ids'],ids) and np.array_equal(data['y'],y)
    output=[]
    for cohort,start in [('development',0),('previously_unused',25000)]:
        key='s'+str(seed)+'_'+selector+'_quadratic_'+cohort
        path=RUN/(key+'_comparison.json')
        if path.exists():output.append(read(path));continue
        rng=np.random.default_rng(20261014^int(hashlib.sha256(key.encode()).hexdigest()[:16],16))
        result=paired_bootstrap(y[start:],q['pred'][start:],c['pred'][start:],rng,plan['draws'])
        result.update(key=key,seed=seed,selection=selector,cohort=cohort,control_family='quadratic',
                      quantum_proof={'name':reference['name'],'checkpoint_sha256':reference['checkpoint_sha256'],'predictions_sha256':sha(qpath)},
                      control_proof={'name':row['name'],'checkpoint_sha256':row['checkpoint_sha256'],'predictions_sha256':sha(cpath)},
                      plan_sha256=sha(RUN/'plan.json'))
        write(path,result)
        output.append(result)
    return output


def watch(plan):
    for name,digest in plan['data_sha256'].items():assert sha(EXP/name)==digest
    assert sha(ROOT/'paper_evidence_references.py')==plan['bootstrap_helper_sha256']
    assert sha(BASE/'development/plan.json')==plan['reference_development_plan_sha256']
    assert sha(BASE/'followup_seed44_seed45/evaluation/plan.json')==plan['seed45_reference_evaluation_plan_sha256']
    x=np.load(EXP/'x.npy')[10000:]
    y=np.load(EXP/'y.npy')[10000:]
    ids=np.load(EXP/'cohort_ids.npz')['pooled'][10000:]
    assert x.shape==(65631,359) and len(np.unique(ids))==65631
    assert not np.intersect1d(ids,np.load(neural.DATA/'val_ids.npy')).size
    assert not np.intersect1d(ids,np.load(neural.DATA/'fit_index.npy')).size
    assert not np.intersect1d(ids,np.load(ROOT/'data/confirmation2_ids.npz')['ids']).size
    count=np.bincount(np.load(neural.DATA/'fit_y.npy'),minlength=10)
    logweights=np.log(count.sum()/(10*count))
    state={'status':'watching','pid':os.getpid(),'plan_sha256':sha(RUN/'plan.json'),'failed':{},'rows':[],'comparisons':[],'missing_references':[]}
    while True:
        training=read(TRAIN/'state.json') if (TRAIN/'state.json').exists() else {}
        for job in read(TRAIN/'plan.json')['jobs']:
            if not (TRAIN/'models'/job['name']/'protocol_audit.json').exists():continue
            for selector,prefix in ENDPOINTS:
                key=job['name']+'_'+selector
                if key in state['failed']:continue
                try:evaluate(job,selector,prefix,x,y,ids,logweights,plan,state)
                except Exception:state['failed'][key]=traceback.format_exc()
        state['rows']=[read(path) for path in sorted(RUN.glob('*_result.json'))]
        missing=[]
        for row in state['rows']:
            try:
                result=compare(row,ids,y,plan)
                if result is None:missing.append('quantum_s'+str(row['seed'])+'_'+row['selection'])
            except Exception:state['failed']['comparison_'+row['name']+'_'+row['selection']]=traceback.format_exc()
        state['comparisons']=[read(path) for path in sorted(RUN.glob('*_comparison.json'))]
        state['missing_references']=sorted(set(missing))
        prior=read(BASE/'followup_seed44_seed45/evaluation/state.json') if (BASE/'followup_seed44_seed45/evaluation/state.json').exists() else {}
        if training.get('status') in ('complete','finished_with_failures') and prior.get('status') in ('complete','finished_with_failures'):
            state['status']='complete' if len(state['rows'])==4 and len(state['comparisons'])==8 and not state['failed'] else 'finished_with_failures_or_missing_references'
        write(RUN/'state.json',state)
        write(RUN/'summary.json',{key:value for key,value in state.items() if key not in ('pid','stage','current','pass_number')})
        if state['status']!='watching':return state
        time.sleep(30)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('freeze','launch','watch','status'))
    parser.add_argument('--expected-train-plan')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    plan=freeze(args.expected_train_plan)
    if args.command=='freeze':out={'status':'frozen','plan_sha256':sha(RUN/'plan.json'),'expected_rows':4,'expected_comparisons':8}
    elif args.command=='watch':out=watch(plan)
    elif args.command=='status':
        state=read(RUN/'state.json') if (RUN/'state.json').exists() else {}
        proc=Path('/proc')/str(state.get('pid'))/'cmdline'
        out={'state':{key:value for key,value in state.items() if key not in ('rows','comparisons')},
             'alive':proc.exists() and 'paper_evidence_fullinteraction_evaluation.py' in proc.read_bytes().decode(),
             'summary':read(RUN/'summary.json') if (RUN/'summary.json').exists() else {},'plan_sha256':sha(RUN/'plan.json')}
    else:
        receipt=read(RUN/'launch.json') if (RUN/'launch.json').exists() else {}
        prior_state=read(RUN/'state.json') if (RUN/'state.json').exists() else {}
        proc=Path('/proc')/str(receipt.get('pid'))/'cmdline'
        if prior_state.get('status') in ('complete','finished_with_failures_or_missing_references'):
            out={'status':'already_finished','plan_sha256':sha(RUN/'plan.json'),'state':prior_state}
        elif proc.exists() and 'paper_evidence_fullinteraction_evaluation.py' in proc.read_bytes().decode():out=dict(receipt,already_running=True)
        else:
            with (RUN/'evaluation.log').open('ab',buffering=0) as log:
                child=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'watch'],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
            out={'status':'launched','pid':child.pid,'plan_sha256':sha(RUN/'plan.json')}
            write(RUN/'launch.json',out)
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()
