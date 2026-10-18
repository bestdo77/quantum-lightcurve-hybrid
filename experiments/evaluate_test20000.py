"""Reproduce all reported metrics and paired intervals on one fixed test cohort."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def score_cm(cm):
    tp=np.diagonal(cm,axis1=-2,axis2=-1)
    den=cm.sum(-2)+cm.sum(-1)
    f=np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0)
    return f.mean(-1),tp.sum(-1)/cm.sum((-2,-1)),f
def metric(y,p):
    cm=np.bincount(y*10+p,minlength=100).reshape(10,10)
    f,a,classes=score_cm(cm)
    return dict(objects=len(y),correct=int(np.trace(cm)),macro_f1=float(f),accuracy=float(a),class_f1=classes.tolist(),confusion_matrix=cm.tolist())
def bootstrap(y,q,c,key,draws=10000):
    # Keep the original subset-reanalysis RNG rule, including the first two rows.
    rngkey=key.replace('_test20000','_extra20000')
    seed=20261018^int(hashlib.sha256(rngkey.encode()).hexdigest()[:16],16)
    rng=np.random.default_rng(seed)
    joint=np.bincount(y*100+q*10+c,minlength=1000)
    fvalues=[];avalues=[]
    for begin in range(0,draws,500):
        sampled=rng.multinomial(len(y),joint/len(y),size=min(500,draws-begin)).reshape(-1,10,10,10)
        qf,qa,_=score_cm(sampled.sum(3));cf,ca,_=score_cm(sampled.sum(2))
        fvalues.extend(qf-cf);avalues.extend(qa-ca)
    return dict(rng_seed=seed,draws=draws,interval95={
        'macro_f1':np.quantile(fvalues,[.025,.975]).tolist(),
        'accuracy':np.quantile(avalues,[.025,.975]).tolist()})
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def evaluate(root,predictions,output=None):
    root=Path(root);output=Path(output) if output else root/'results'
    catalog=json.loads((root/'data/test_model_catalog.json').read_text())
    pairs=json.loads((root/'plans/test20000_pairs.json').read_text())
    with np.load(predictions,allow_pickle=False) as z:
        arrays={key:z[key].copy() for key in z.files}
    ids,y=arrays.pop('ids'),arrays.pop('y')
    assert len(ids)==len(np.unique(ids))==len(y)==20000
    proof=dict(archive_sha256=digest(predictions),ids_array_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),labels_array_sha256=hashlib.sha256(y.tobytes()).hexdigest())
    plan=json.loads((root/'plans/test20000_evaluation.json').read_text())
    assert proof['archive_sha256']==plan['predictions_archive_sha256']
    assert proof['ids_array_sha256']==plan['ids_array_sha256']
    assert proof['labels_array_sha256']==plan['labels_array_sha256']
    rows=[];computed={}
    for key,metadata in sorted(catalog.items()):
        pred=arrays[key].astype(np.int64)
        assert pred.shape==y.shape and np.isin(pred,np.arange(10)).all()
        assert hashlib.sha256(arrays[key].tobytes()).hexdigest()==metadata['test_prediction_array_sha256']
        measured=metric(y,pred)
        assert all(abs(measured[k]-metadata['metrics'][k])<1e-12 for k in ('macro_f1','accuracy'))
        record={k:v for k,v in metadata.items() if k!='metrics'}
        record.update(test_objects=20000,cohort='test20000',metrics=measured,predictions_proof=proof)
        write(output/'audited'/(key+'_result.json'),record);computed[key]=record
        rows.append(dict(name=key,kind=metadata.get('kind'),seed=metadata.get('seed'),selection=metadata.get('selection'),parameters=metadata.get('parameters'),test_objects=20000,test_macro_f1=measured['macro_f1'],test_accuracy=measured['accuracy']))
    compared=[]
    for index,pair in enumerate(pairs):
        qkey,ckey=pair['quantum'],pair['control'];q=arrays[qkey].astype(np.int64);c=arrays[ckey].astype(np.int64)
        qm,cm=computed[qkey]['metrics'],computed[ckey]['metrics']
        record={**pair,'objects':20000,'cohort':'test20000','quantum_metrics':qm,'control_metrics':cm,
                'quantum_minus_control':{k:qm[k]-cm[k] for k in ('macro_f1','accuracy')},
                **bootstrap(y,q,c,pair['key']),
                'quantum_only_correct':int(np.sum((q==y)&(c!=y))),
                'control_only_correct':int(np.sum((q!=y)&(c==y))),
                'quantum_proof':{k:computed[qkey][k] for k in ('source_predictions_sha256','test_prediction_array_sha256')},
                'control_proof':{k:computed[ckey][k] for k in ('source_predictions_sha256','test_prediction_array_sha256')},
                'interval_scope':'Fixed fitted models; exploratory paired object bootstrap; no training-seed, search or multiplicity uncertainty.'}
        write(output/'comparisons'/(pair['key']+'_comparison.json'),record);compared.append(record)
        if index%10==0:print('Completed paired comparisons',index+1,'/',len(pairs),flush=True)
    for filename,data in [('main_results.csv',rows),('paired_comparisons.csv',[
            dict(key=r['key'],seed=r.get('seed'),control_family=r['control_family'],selection=r['selection'],test_objects=20000,
                 delta_f1_pp=100*r['quantum_minus_control']['macro_f1'],f1_low_pp=100*r['interval95']['macro_f1'][0],f1_high_pp=100*r['interval95']['macro_f1'][1],
                 delta_accuracy_pp=100*r['quantum_minus_control']['accuracy'],accuracy_low_pp=100*r['interval95']['accuracy'][0],accuracy_high_pp=100*r['interval95']['accuracy'][1]) for r in compared])]:
        path=output/'tables'/filename;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
    write(output/'test20000_summary.json',dict(test_objects=20000,models=len(rows),comparisons=len(compared),predictions_proof=proof,confirmation_accessed=False))
    print('Verified',len(rows),'model rows and',len(compared),'paired comparisons on exactly 20,000 objects.',flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--predictions',type=Path,required=True);parser.add_argument('--output',type=Path)
    args=parser.parse_args();evaluate(args.root,args.predictions,args.output)
