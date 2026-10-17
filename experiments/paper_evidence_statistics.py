"""Predeclared paired-object intervals and separate fine-tuning seed summaries."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import paper_evidence_neural as neural
import paper_evidence_tree_selection as trees
from paper_evidence_references import paired_bootstrap
from train_priority_14235 import metric, sha, write

ROOT=Path(__file__).resolve().parent
DEV=neural.RUN.parent/'development'
RUN=neural.RUN.parent/'statistics'
EXP=ROOT/'expanded_evaluation_20261014'
NEURAL_PLAN='7a94001fe61846f105a460cedb3804673f26a7e8b1bb4ceab13cb13c38e72ee7'
DEV_PLAN='293eef107c3ae584c50dd1bebbd1b298d826312562fa3e992df1f0145ccd781f'
TREE_PLAN='27bbc0e956d5818550db3e083c31e96f188af7f14b7881e2caa4e8e95d13dffd'


def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())


def freeze():
    RUN.mkdir(parents=True,exist_ok=True)
    assert sha(neural.PLAN)==NEURAL_PLAN and sha(DEV/'plan.json')==DEV_PLAN and sha(trees.RUN/'plan.json')==TREE_PLAN
    path=RUN/'plan.json'
    if path.exists():
        plan=read(path)
        assert plan['source_sha256']==sha(__file__) and plan['bootstrap_helper_sha256']==sha(ROOT/'paper_evidence_references.py')
        return plan
    plan={'status':'frozen_before_new_quantum_comparison','created_utc':now(),'source_sha256':sha(__file__),
          'neural_plan_sha256':NEURAL_PLAN,'development_plan_sha256':DEV_PLAN,'tree_plan_sha256':TREE_PLAN,
          'bootstrap_helper_sha256':sha(ROOT/'paper_evidence_references.py'),'draws':10000,
          'seed_rule':'20261014 xor first16hex(SHA256(comparison_key)); independent of job completion order.',
          'comparisons':'For three seeds:quantum vs validation-selected pureMLP,matchedMLP,reselectedRF,reselectedLightGBM. Also seed42 vs noCNOT,frozenQuantum,strongMLP,quadratic. Both selectors and both nested development scopes.',
          'expected_comparisons':64,'cohorts':{'development':0,'previously_unused':25000},
          'interval_scope':'Pointwise paired-object 95% percentile intervals, conditional on each frozen model pair. No search uncertainty or multiplicity correction; exploratory.',
          'seed_summary':'Mean and sample SD of paired differences across three conditional fine-tuning seeds. Object n is not multiplied by3; no CI for seed mean inferred from object CIs.',
          'confirmation_accessed':False}
    write(path,plan);return plan


def neural_prediction(row,ids,y):
    key=row['name']+'_'+row['selection']
    path=DEV/(key+'_predictions.npz')
    assert row['status']=='passed' and row['plan_sha256']==DEV_PLAN and row['complete_reload_prediction_disagreements']==0
    assert sha(path)==row['predictions_sha256']
    with np.load(path) as array:
        assert np.array_equal(array['ids'],ids) and np.array_equal(array['y'],y)
        pred=array['pred'].copy()
    scores=metric(y,pred)
    assert all(abs(scores[m]-row['metrics']['development'][m])<1e-12 for m in scores)
    return pred,{'name':row['name'],'selection':row['selection'],'checkpoint_sha256':row['checkpoint_sha256'],
                 'predictions_sha256':sha(path),'development_result_sha256':sha(DEV/(key+'_result.json'))}


def tree_prediction(selection,result,ids,y):
    row=next(r for r in result['all_development_rows'] if r['name']==selection['name'] and r['alpha']==selection['alpha'])
    assert row['complete_reload_max_error']==row['old_three_split_prediction_disagreements']==0
    path=trees.RUN/(selection['name']+'_alpha'+str(selection['alpha'])+'_selected_predictions.npz')
    with np.load(path) as array:
        assert np.array_equal(array['ids'],ids) and np.array_equal(array['y'],y)
        pred=array['pred'].copy()
    scores=metric(y,pred)
    assert all(abs(scores[m]-row['metrics']['development'][m])<1e-12 for m in scores)
    return pred,{'name':selection['name'],'selection':selection['selection'],'alpha':selection['alpha'],
                 'checkpoint_sha256':row['checkpoint_sha256'],'predictions_sha256':sha(path),
                 'tree_selection_sha256':result['selection_sha256']}


def seed_summary(results):
    summaries=[]
    for family in ('pure_mlp','matched_mlp','rf','lightgbm'):
        for selector in ('best_macro','best_acc'):
            for cohort in ('development','previously_unused'):
                group=[r for r in results if r['control_family']==family and r['selection']==selector and r['cohort']==cohort]
                if {r['seed'] for r in group}!={42,43,44} or len(group)!=3:continue
                values={m:[r['quantum_minus_control'][m] for r in sorted(group,key=lambda r:r['seed'])] for m in ('macro_f1','accuracy')}
                summaries.append({'control_family':family,'selection':selector,'cohort':cohort,'seeds':[42,43,44],
                     'objects_each':group[0]['objects'],'independent_M8_pretrainings':1,
                     'mean_quantum_minus_control':{m:float(np.mean(v)) for m,v in values.items()},
                     'sample_sd':{m:float(np.std(v,ddof=1)) for m,v in values.items()},'paired_seed_values':values})
    return summaries


def observe(plan):
    y=np.load(EXP/'y.npy')[10000:];ids=np.load(EXP/'cohort_ids.npz')['pooled'][10000:]
    assert len(ids)==len(np.unique(ids))==65631
    results=[read(path) for path in sorted(RUN.glob('*_comparison.json'))]
    done={r['key'] for r in results}
    state={'status':'running','pid':os.getpid(),'started_utc':now(),'completed':sorted(done),'failed':{},'plan_sha256':sha(RUN/'plan.json')}
    while True:
        dev=read(DEV/'summary.json') if (DEV/'summary.json').exists() else {}
        rows=dev.get('rows',[])
        tree=read(trees.RUN/'result.json') if (trees.RUN/'result.json').exists() else None
        pairs=[]
        for seed in (42,43,44):
            for selector in ('best_macro','best_acc'):
                quantum=next((r for r in rows if r['kind']=='quantum' and r['seed']==seed and r['selection']==selector),None)
                if quantum is None:continue
                for family in ('matched_mlp','pure_mlp','no_cnot','frozen_quantum','strong_mlp','quadratic'):
                    if family in ('no_cnot','frozen_quantum','strong_mlp','quadratic') and seed!=42:continue
                    group=[r for r in rows if r['kind']==family and r['seed']==seed and r['selection']==selector]
                    if family=='pure_mlp':
                        if len(group)!=2:continue
                        first,second=('macro_f1','accuracy') if selector=='best_macro' else ('accuracy','macro_f1')
                        control=max(group,key=lambda r:(r['validation'][first],r['validation'][second]))
                    else:
                        if len(group)!=1:continue
                        control=group[0]
                    pairs.append((seed,selector,family,quantum,control,False))
                if tree:
                    assert tree['status']=='complete' and tree['plan_sha256']==TREE_PLAN
                    for family in ('rf','lightgbm'):
                        control=next(r for r in tree['selected'] if r['kind']==family and r['selection']==selector)
                        pairs.append((seed,selector,family,quantum,control,True))
        for seed,selector,family,qrow,crow,is_tree in pairs:
            todo=[]
            for cohort,start in plan['cohorts'].items():
                key='s'+str(seed)+'_'+selector+'_'+family+'_'+cohort
                if key not in done and key not in state['failed']:todo.append((key,cohort,start))
            if not todo:continue
            try:
                qp,qproof=neural_prediction(qrow,ids,y)
                cp,cproof=tree_prediction(crow,tree,ids,y) if is_tree else neural_prediction(crow,ids,y)
                for key,cohort,start in todo:
                    state.update(stage='paired_bootstrap',current=key,checked_utc=now());write(RUN/'state.json',state)
                    seed_value=20261014^int(hashlib.sha256(key.encode()).hexdigest()[:16],16)
                    result=paired_bootstrap(y[start:],qp[start:],cp[start:],np.random.default_rng(seed_value),plan['draws'])
                    result.update(key=key,seed=seed,selection=selector,control_family=family,cohort=cohort,
                                  quantum_proof=qproof,control_proof=cproof,plan_sha256=sha(RUN/'plan.json'))
                    write(RUN/(key+'_comparison.json'),result);results.append(result);done.add(key)
                    state['completed']=sorted(done)
            except Exception as error:
                for key,cohort,start in todo:state['failed'][key]={'error':repr(error),'traceback':traceback.format_exc()}
            write(RUN/'summary.json',{'status':'running','comparisons':results,'seed_summaries':seed_summary(results),'plan_sha256':sha(RUN/'plan.json')})
            write(RUN/'state.json',state)
        tree_state=read(trees.RUN/'state.json') if (trees.RUN/'state.json').exists() else {}
        terminal=dev.get('status') in ('complete','finished_with_failures') and tree_state.get('status') in ('complete','failed')
        if len(done)==plan['expected_comparisons'] or terminal:
            state['status']='complete' if len(done)==plan['expected_comparisons'] and not state['failed'] else 'finished_with_failures'
            write(RUN/'summary.json',{'status':state['status'],'comparisons':results,'seed_summaries':seed_summary(results),
                                     'plan_sha256':sha(RUN/'plan.json'),'confirmation_accessed':False,'finished_utc':now()})
            write(RUN/'state.json',dict(state,checked_utc=now()));return state
        state.update(stage='waiting_for_frozen_predictions',current=None,checked_utc=now());write(RUN/'state.json',state)
        time.sleep(30)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','launch','observe','status'))
    args=parser.parse_args();plan=freeze()
    if args.command=='freeze':out={'status':'frozen','plan_sha256':sha(RUN/'plan.json'),'comparisons':plan['expected_comparisons']}
    elif args.command=='observe':out=observe(plan)
    elif args.command=='status':
        state=read(RUN/'state.json') if (RUN/'state.json').exists() else {}
        proc=Path('/proc')/str(state.get('pid'))/'cmdline'
        out={'state':state,'alive':proc.exists() and Path(__file__).name in proc.read_bytes().decode(),
             'summary':read(RUN/'summary.json') if (RUN/'summary.json').exists() else None}
    else:
        receipt=read(RUN/'launch.json') if (RUN/'launch.json').exists() else {}
        proc=Path('/proc')/str(receipt.get('pid'))/'cmdline'
        if proc.exists() and Path(__file__).name in proc.read_bytes().decode():out=dict(receipt,already_running=True)
        elif (RUN/'state.json').exists() and read(RUN/'state.json')['status']=='complete':out={'status':'complete'}
        else:
            with (RUN/'run.log').open('ab',buffering=0) as log:
                process=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'observe'],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
            out={'status':'launched','pid':process.pid,'utc':now(),'plan_sha256':sha(RUN/'plan.json')};write(RUN/'launch.json',out)
    print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
