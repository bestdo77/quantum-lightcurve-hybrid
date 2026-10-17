"""Frozen RF extension and paired object-bootstrap evidence for existing models."""
import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import binomtest

from train_priority_14235 import DATA, ROOT, metric, sha, write

RUN = Path('/mnt/hdd/yxh/qml_paper_evidence_20261014/reference')
EXP = ROOT / 'expanded_evaluation_20261014'
RF = ROOT / 'rf_full_reference_20261012'


def now():
    return datetime.now(timezone.utc).isoformat()


def freeze():
    RUN.mkdir(parents=True, exist_ok=True)
    path = RUN / 'plan.json'
    if path.exists():
        plan = json.loads(path.read_text())
        assert plan['source_sha256'] == sha(__file__)
        return plan
    complete = json.loads((RF / 'complete.json').read_text())
    audit = json.loads((RF / 'completion_audit.json').read_text())
    assert complete['status'] == 'complete' and audit['status'] == 'passed'
    assert all(row['status'] == 'passed' for row in audit['models'])
    selected = complete['selected']
    model = next(row for row in complete['results'] if row['name'] == selected['name'])
    assert model['trees'] == 500 and sha(model['model_path']) == model['model_sha256']
    plan = {'status': 'frozen_before_RF_new_scores', 'created_utc': now(), 'rf': model,
            'rf_alpha': selected['alpha'], 'selection_rule': 'Original validation-selected RF model and alpha, no new selection.',
            'expanded_summary_sha256': sha(EXP / 'results_summary.json'),
            'expanded_plan_sha256': sha(EXP / 'plan.json'), 'source_sha256': sha(__file__),
            'bootstrap_draws': 10000, 'bootstrap_seed': 20261014,
            'bootstrap_unit': 'Paired astronomical object: true label and both model predictions resampled together. Unique object IDs verified.',
            'bootstrap_method': 'Ordinary paired nonparametric object bootstrap, using a multinomial over the sufficient joint counts (y,pred_Q,pred_control); class proportions fluctuate.',
            'interval': 'Pointwise 95% percentile interval, conditional on the frozen trained models; no training-seed or search uncertainty.',
            'inference_scope': 'Exploratory analysis of already evaluated development cohorts. Model-conditional uncertainty is not independent multi-seed replication.',
            'mcnemar': 'Two-sided exact binomial test on discordant correct/incorrect pairs; ACC endpoint only, exploratory p-values reported without confirmatory significance declarations.',
            'confirmation2_evaluated': False}
    write(path, plan)
    return plan


def forest_probabilities(model, x, state, repetition):
    assert len(model.estimators_) == 500 and np.array_equal(model.classes_, np.arange(10))
    result = np.zeros((len(x), 10), dtype=np.float64)
    for index, tree in enumerate(model.estimators_):
        probabilities = tree.predict_proba(x, check_input=False)
        assert probabilities.shape == result.shape
        result += probabilities
        if (index + 1) % 50 == 0:
            state.update(stage='RF_prediction', repetition=repetition, trees=index + 1, checked_utc=now())
            write(RUN / 'state.json', state)
    return result / len(model.estimators_)


def confusion_scores(cm):
    diagonal = np.diagonal(cm, axis1=-2, axis2=-1)
    denominator = cm.sum(-2) + cm.sum(-1)
    f1 = np.divide(2*diagonal, denominator, out=np.zeros_like(diagonal, dtype=float), where=denominator > 0)
    return f1.mean(-1), diagonal.sum(-1) / cm.sum((-2, -1))


def paired_bootstrap(y, qp, cp, rng, draws):
    joint = np.bincount(y*100 + qp*10 + cp, minlength=1000).reshape(10,10,10)
    total = len(y)
    sampled = rng.multinomial(total, joint.reshape(-1)/total, size=draws).reshape(draws,10,10,10)
    qcm, ccm = sampled.sum(3), sampled.sum(2)
    qf, qa = confusion_scores(qcm)
    cf, ca = confusion_scores(ccm)
    true_q, true_c = qp == y, cp == y
    q_only = int(np.count_nonzero(true_q & ~true_c))
    c_only = int(np.count_nonzero(~true_q & true_c))
    qmetric, cmetric = metric(y,qp), metric(y,cp)
    return {'objects': total, 'quantum': qmetric, 'control': cmetric,
            'quantum_minus_control': {key:qmetric[key]-cmetric[key] for key in qmetric},
            'interval95': {'macro_f1': np.quantile(qf-cf,[.025,.975]).tolist(),
                           'accuracy': np.quantile(qa-ca,[.025,.975]).tolist()},
            'quantum_only_correct': q_only, 'control_only_correct': c_only,
            'exact_mcnemar_accuracy_p': float(binomtest(q_only, q_only+c_only, .5).pvalue) if q_only+c_only else 1.0,
            'draws': draws}


def execute(plan):
    state = {'status':'running','pid':os.getpid(),'started_utc':now(),'plan_sha256':sha(RUN/'plan.json')}
    write(RUN/'state.json',state)
    try:
        assert sha(EXP/'results_summary.json') == plan['expanded_summary_sha256']
        assert sha(plan['rf']['model_path']) == plan['rf']['model_sha256']
        x, y = np.load(EXP/'x.npy'), np.load(EXP/'y.npy')
        ids = np.load(EXP/'cohort_ids.npz')['pooled']
        model = joblib.load(plan['rf']['model_path'])
        probabilities = forest_probabilities(model,x,state,1)
        freq = np.bincount(np.load(DATA/'fit_y.npy'),minlength=10)
        logweights = np.log(freq.sum()/(10*freq))
        pred = (np.log(np.clip(probabilities,1e-15,1))-plan['rf_alpha']*logweights).argmax(1)
        repeat = forest_probabilities(model,x,state,2)
        assert np.array_equal(probabilities,repeat)
        old_counts = [('val',0,10000),('report',10000,15000),('extra',15000,35000)]
        for split,start,end in old_counts:
            cached = np.load(RF/plan['rf']['name']/(split+'_predictions.npz'))
            assert np.array_equal(ids[start:end],cached['ids'])
            assert np.array_equal(y[start:end],cached['y'])
            assert np.array_equal(pred[start:end],cached['pred'])
        np.savez_compressed(RUN/'rf_predictions.npz',ids=ids,y=y,pred=pred,probabilities=probabilities,
                            alpha=np.asarray(plan['rf_alpha']))
        write(RUN/'rf_result.json',{'status':'passed','name':plan['rf']['name'],
             'metrics':{'pooled':metric(y,pred),'fresh_previous':metric(y[35000:],pred[35000:]),
                        'development_no_validation':metric(y[10000:],pred[10000:])},
             'checkpoint_sha256':sha(plan['rf']['model_path']), 'repeated_probabilities_identical':True,
             'old_three_split_predictions_identical':True,'predictions_sha256':sha(RUN/'rf_predictions.npz')})
        summary = json.loads((EXP/'results_summary.json').read_text())
        results = []
        rng = np.random.default_rng(plan['bootstrap_seed'])
        for selector in ('best_macro','best_acc'):
            q = next(m for m in summary['models'] if m['kind']=='quantum' and m['selection']==selector)
            qp = np.load(EXP/(q['name']+'_'+selector+'_predictions.npz'))['pred']
            controls = [(m['name'], np.load(EXP/(m['name']+'_'+m['selection']+'_predictions.npz'))['pred'])
                        for m in summary['models'] if (m['kind']=='classical' and m['selection']==selector) or m['kind']=='lightgbm']
            controls += [(plan['rf']['name'], pred)]
            for name,cp in controls:
                for cohort,start in (('development_no_validation',10000),('fresh_previous',35000)):
                    state.update(stage='bootstrap',selection=selector,control=name,cohort=cohort,checked_utc=now())
                    write(RUN/'state.json',state)
                    result = paired_bootstrap(y[start:],qp[start:],cp[start:],rng,plan['bootstrap_draws'])
                    result.update(selection=selector,control_name=name,cohort=cohort)
                    results.append(result)
        complete = {'status':'complete','plan':plan,'plan_sha256':sha(RUN/'plan.json'),
                    'rf':json.loads((RUN/'rf_result.json').read_text()),'paired_evidence':results,
                    'finished_utc':now(),'confirmation2_evaluated':False}
        write(RUN/'evidence_summary.json',complete)
        state.update(status='complete',checked_utc=now(),finished_utc=now())
        write(RUN/'state.json',state)
    except Exception as error:
        state.update(status='failed',error=repr(error),traceback=traceback.format_exc(),checked_utc=now())
        write(RUN/'state.json',state)
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('freeze','launch','run','status'))
    args=parser.parse_args();plan=freeze()
    if args.command=='freeze': out={'status':'frozen','plan':plan}
    elif args.command=='run': execute(plan);out={'status':'complete'}
    elif args.command=='status':
        out={'state':json.loads((RUN/'state.json').read_text()) if (RUN/'state.json').exists() else None,
             'summary':json.loads((RUN/'evidence_summary.json').read_text()) if (RUN/'evidence_summary.json').exists() else None}
    else:
        receipt=json.loads((RUN/'launch.json').read_text()) if (RUN/'launch.json').exists() else {}
        pfile=Path('/proc')/str(receipt.get('pid'))/'cmdline'
        if pfile.exists() and Path(__file__).name in pfile.read_bytes().decode():out=dict(receipt,already_running=True)
        elif (RUN/'evidence_summary.json').exists():out={'status':'complete'}
        else:
            with (RUN/'run.log').open('ab',buffering=0) as log:
                p=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'run'],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
            out={'status':'launched','pid':p.pid,'utc':now(),'plan_sha256':sha(RUN/'plan.json')};write(RUN/'launch.json',out)
    print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
