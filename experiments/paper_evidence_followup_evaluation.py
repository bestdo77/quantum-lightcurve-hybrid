"""Frozen CPU evaluation and paired statistics for the immutable supplement."""
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import paper_evidence_neural as neural
import paper_evidence_development as original_development
import paper_evidence_tree_selection as trees
from paper_evidence_references import paired_bootstrap
from train_priority_14235 import predict, sha, write
from evaluate_expanded_p1 import metrics

TRAIN = neural.RUN.parent / 'followup_seed44_seed45'
RUN = TRAIN / 'evaluation'
OLD_DEV = neural.RUN.parent / 'development'
EXP = ROOT / 'expanded_evaluation_20261014'
EXPECTED = '8fce3b618bb8ed92015825019aa3f21902c66f52f658c5ca50b3e8ef5604ce48'
ENDPOINTS = [('best_macro','best_macro'),('best_acc','best_accuracy')]


def read(path): return json.loads(Path(path).read_text())


def freeze():
    RUN.mkdir(parents=True,exist_ok=True)
    assert sha(TRAIN/'plan.json') == EXPECTED
    if (RUN/'plan.json').exists():
        plan = read(RUN/'plan.json')
        assert plan['source_sha256'] == sha(__file__)
        return plan
    old_plan = read(OLD_DEV/'plan.json')
    plan = {'status':'frozen_before_supplement_performance','source_sha256':sha(__file__),
            'supplement_train_plan_sha256':EXPECTED,'original_development_plan_sha256':sha(OLD_DEV/'plan.json'),
            'tree_plan_sha256':sha(trees.RUN/'plan.json'),'data_sha256':old_plan['data_sha256'],
            'original_statistics_plan_sha256':sha(neural.RUN.parent/'statistics/plan.json'),
            'bootstrap_helper_sha256':sha(ROOT/'paper_evidence_references.py'),'draws':10000,
            'inference':'CPU FP32, 4threads, batch256; complete independent reload repeat for both endpoints.',
            'comparisons':'seed44quantum vs seed44frozen/noCNOT; seed45quantum vs seed45matched/pureMLP and fixed reselectedRF/LightGBM; both endpoints and nested scopes.',
            'expected_comparisons':24,'pureMLP_selection':'Two predeclared LRs selected only on105297 validation per endpoint.',
            'interval_scope':'Pointwise exploratory paired-object percentile intervals, fixed model pairs, no search uncertainty or multiplicity correction.',
            'seed_summary':'All attempts42/43/44/45 disclosed; original43 excluded after numerical audit failure. Descriptive mean/sampleSD of audited42/44/45, conditional on shared M8.',
            'confirmation_accessed':False}
    write(RUN/'plan.json',plan)
    return plan


def evaluate(job,selector,prefix,x,y,ids,logweights,state):
    key = job['name']+'_'+selector
    destination = RUN/(key+'_result.json')
    if destination.exists(): return read(destination)
    folder = TRAIN/'models'/job['name']
    audit,protocol = read(folder/'result_audit.json'),read(folder/'protocol_audit.json')
    assert audit['status'] == protocol['status'] == 'passed'
    assert audit['plan_sha256'] == EXPECTED and protocol['result_audit_sha256'] == sha(folder/'result_audit.json')
    selected = audit[selector]
    checkpoint = folder/(prefix+'.pt')
    assert sha(checkpoint) == selected['checkpoint_sha256']
    raws = []
    for index in (1,2):
        state.update(stage='CPU_inference',current=key,pass_number=index)
        write(RUN/'state.json',state)
        model = neural.make_model(job['spec']).eval()
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
        raws.append(predict(model,torch.from_numpy(x),batch=256))
    predictions = [(raw-selected['alpha']*logweights).argmax(1) for raw in raws]
    assert np.array_equal(predictions[0],predictions[1]),key
    pred = predictions[0]
    old_checks = {}
    for split,start,end in [('report',0,5000),('extra',5000,25000)]:
        cached = np.load(folder/(split+'_'+prefix+'_predictions.npz'))
        assert np.array_equal(cached['ids'],ids[start:end]) and np.array_equal(cached['y'],y[start:end])
        assert np.array_equal(cached['pred'],pred[start:end]),(key,split)
        old_checks[split] = {'objects':end-start,'prediction_disagreements':0}
    prediction_path = RUN/(key+'_predictions.npz')
    np.savez_compressed(prediction_path,ids=ids,y=y,pred=pred,logits=raws[0],alpha=np.asarray(selected['alpha']))
    row = {'status':'passed','name':job['name'],'kind':job['spec']['kind'],'seed':job['spec']['seed'],
           'learning_rate':job['learning_rate'],'selection':selector,'epoch':selected['epoch'],'alpha':selected['alpha'],
           'parameters':audit['counts']['total'],'validation':selected['metrics']['validation'],
           'metrics':{'development':metrics(y,pred),'previously_unused':metrics(y[25000:],pred[25000:])},
           'checkpoint_sha256':sha(checkpoint),'predictions_sha256':sha(prediction_path),
           'old_prediction_checks':old_checks,'complete_reload_prediction_disagreements':0,
           'plan_sha256':sha(RUN/'plan.json'),'training_result_audit_sha256':sha(folder/'result_audit.json'),
           'protocol_audit_sha256':sha(folder/'protocol_audit.json'),'confirmation_accessed':False}
    write(destination,row)
    print(json.dumps({'evaluated':key,'metrics':row['metrics']}),flush=True)
    return row


def get_predictions(row,ids,y):
    folder = OLD_DEV if row.get('plan_sha256') == sha(OLD_DEV/'plan.json') else RUN
    path = folder/(row['name']+'_'+row['selection']+'_predictions.npz')
    assert sha(path) == row['predictions_sha256']
    data = np.load(path)
    assert np.array_equal(data['ids'],ids) and np.array_equal(data['y'],y)
    return data['pred'],{'name':row['name'],'checkpoint_sha256':row['checkpoint_sha256'],'predictions_sha256':sha(path)}


def compare(plan,rows,ids,y,state):
    original = read(OLD_DEV/'summary.json')['rows']
    tree_result = read(trees.RUN/'result.json')
    done = {r['key'] for r in state['comparisons']}
    pairs = []
    for selector,prefix in ENDPOINTS:
        q44 = next(r for r in original if r['name']=='quantum_s44' and r['selection']==selector)
        for kind in ('frozen_quantum','no_cnot'):
            control = next((r for r in rows if r['kind']==kind and r['selection']==selector),None)
            if control: pairs.append((44,selector,kind,q44,control,False))
        q45 = next((r for r in rows if r['name']=='quantum_s45' and r['selection']==selector),None)
        if not q45: continue
        for kind in ('matched_mlp','pure_mlp'):
            group = [r for r in rows if r['kind']==kind and r['selection']==selector]
            if len(group) != (2 if kind=='pure_mlp' else 1): continue
            main,other = ('macro_f1','accuracy') if selector=='best_macro' else ('accuracy','macro_f1')
            chosen = max(group,key=lambda r:(r['validation'][main],r['validation'][other]))
            pairs.append((45,selector,kind,q45,chosen,False))
        for family in ('rf','lightgbm'):
            chosen = next(r for r in tree_result['selected'] if r['kind']==family and r['selection']==selector)
            pairs.append((45,selector,family,q45,chosen,True))
    for seed,selector,family,qrow,crow,is_tree in pairs:
        if all('s'+str(seed)+'_'+selector+'_'+family+'_'+cohort in done for cohort in ('development','previously_unused')): continue
        qp,qproof = get_predictions(qrow,ids,y)
        if is_tree:
            path = trees.RUN/(crow['name']+'_alpha'+str(crow['alpha'])+'_selected_predictions.npz')
            data = np.load(path)
            assert np.array_equal(data['ids'],ids) and np.array_equal(data['y'],y)
            cp,cproof = data['pred'],{'name':crow['name'],'alpha':crow['alpha'],'predictions_sha256':sha(path)}
        else: cp,cproof = get_predictions(crow,ids,y)
        for cohort,start in [('development',0),('previously_unused',25000)]:
            key = 's'+str(seed)+'_'+selector+'_'+family+'_'+cohort
            if key in done: continue
            state.update(stage='paired_bootstrap',current=key)
            write(RUN/'state.json',state)
            rng = np.random.default_rng(20261014^int(hashlib.sha256(key.encode()).hexdigest()[:16],16))
            result = paired_bootstrap(y[start:],qp[start:],cp[start:],rng,plan['draws'])
            result.update(key=key,seed=seed,selection=selector,control_family=family,cohort=cohort,
                          quantum_proof=qproof,control_proof=cproof,plan_sha256=sha(RUN/'plan.json'))
            write(RUN/(key+'_comparison.json'),result)
            state['comparisons'].append(result)
            done.add(key)


def watch(plan):
    for name,digest in plan['data_sha256'].items(): assert sha(EXP/name)==digest
    assert sha(ROOT/'paper_evidence_references.py')==plan['bootstrap_helper_sha256']
    assert sha(trees.RUN/'plan.json')==plan['tree_plan_sha256']
    assert sha(neural.RUN.parent/'statistics/plan.json')==plan['original_statistics_plan_sha256']
    x = np.load(EXP/'x.npy')[10000:]
    y = np.load(EXP/'y.npy')[10000:]
    ids = np.load(EXP/'cohort_ids.npz')['pooled'][10000:]
    assert x.shape==(65631,359)
    assert not np.intersect1d(ids,np.load(neural.DATA/'val_ids.npy')).size
    assert not np.intersect1d(ids,np.load(ROOT/'data/confirmation2_ids.npz')['ids']).size
    freq = np.bincount(np.load(neural.DATA/'fit_y.npy'),minlength=10)
    logweights = np.log(freq.sum()/(10*freq))
    state = {'status':'watching','pid':os.getpid(),'plan_sha256':sha(RUN/'plan.json'),
             'rows':[], 'comparisons':[read(p) for p in sorted(RUN.glob('*_comparison.json'))], 'failed':{}}
    jobs = read(TRAIN/'plan.json')['jobs']
    while True:
        training = read(TRAIN/'state.json')
        for job in jobs:
            if not (TRAIN/'models'/job['name']/'protocol_audit.json').exists(): continue
            for selector,prefix in ENDPOINTS:
                key = job['name']+'_'+selector
                if key in state['failed']: continue
                try: evaluate(job,selector,prefix,x,y,ids,logweights,state)
                except Exception:
                    state['failed'][key] = traceback.format_exc()
        state['rows'] = [read(p) for p in sorted(RUN.glob('*_result.json'))]
        try: compare(plan,state['rows'],ids,y,state)
        except Exception:
            state['failed']['comparison'] = traceback.format_exc()
        if training['status'] in ('complete','finished_with_failures'):
            state['status'] = 'complete' if len(state['rows'])==12 and len(state['comparisons'])==24 and not state['failed'] else 'finished_with_failures'
        write(RUN/'state.json',state)
        summary={k:state[k] for k in ('status','rows','comparisons','failed','plan_sha256')}
        original_stats=read(neural.RUN.parent/'statistics/summary.json')['comparisons']
        summaries=[]
        combined=original_stats+state['comparisons']
        for family in ('matched_mlp','pure_mlp','rf','lightgbm'):
            for selector,prefix in ENDPOINTS:
                for cohort in ('development','previously_unused'):
                    group=[r for r in combined if r['control_family']==family and r['selection']==selector
                           and r['cohort']==cohort and r['seed'] in (42,44,45)]
                    if {r['seed'] for r in group}!={42,44,45} or len(group)!=3: continue
                    values={m:[r['quantum_minus_control'][m] for r in sorted(group,key=lambda r:r['seed'])]
                            for m in ('macro_f1','accuracy')}
                    summaries.append({'control_family':family,'selection':selector,'cohort':cohort,
                                      'audited_seeds':[42,44,45],'all_attempted_seeds':[42,43,44,45],
                                      'excluded_seed43':'Original strict numerical audit failed; selected outcomes independently reproduced but original status preserved.',
                                      'objects_each':group[0]['objects'],'independent_M8_pretrainings':1,
                                      'mean_quantum_minus_control':{m:float(np.mean(v)) for m,v in values.items()},
                                      'sample_sd':{m:float(np.std(v,ddof=1)) for m,v in values.items()},
                                      'paired_seed_values':values,'mean_difference_CI_computed':False})
        summary['seed_summaries']=summaries
        write(RUN/'summary.json',summary)
        if state['status'] != 'watching': return state
        time.sleep(20)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('freeze','launch','watch','status'))
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    plan=freeze()
    if args.command=='freeze': out={'status':'frozen','plan_sha256':sha(RUN/'plan.json'),'expected_comparisons':24}
    elif args.command=='watch': out=watch(plan)
    elif args.command=='status':
        state=read(RUN/'state.json') if (RUN/'state.json').exists() else {}
        proc=Path('/proc')/str(state.get('pid'))/'cmdline'
        alive=proc.exists() and 'paper_evidence_followup_evaluation.py' in proc.read_bytes().decode()
        out={'state':{k:v for k,v in state.items() if k not in ('rows','comparisons')},'alive':alive,
             'summary':read(RUN/'summary.json') if (RUN/'summary.json').exists() else {},'plan_sha256':sha(RUN/'plan.json')}
    else:
        receipt=read(RUN/'launch.json') if (RUN/'launch.json').exists() else {}
        proc=Path('/proc')/str(receipt.get('pid'))/'cmdline'
        if proc.exists() and 'paper_evidence_followup_evaluation.py' in proc.read_bytes().decode(): out=dict(receipt,already_running=True)
        else:
            with (RUN/'evaluation.log').open('ab',buffering=0) as log:
                child=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'watch'],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
            out={'status':'launched','pid':child.pid,'plan_sha256':sha(RUN/'plan.json')}
            write(RUN/'launch.json',out)
    print(json.dumps(out),flush=True)


if __name__=='__main__': main()
