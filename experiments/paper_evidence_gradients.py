"""Training-only gradient diagnostics, independent physical differentiation.

This does not train or select a model. It distinguishes zero residual-head
initialization from a locally flat circuit and benchmarks CPU simulation cost.
All data are from the fitting partition; sealed outcomes are never opened.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import paper_evidence_neural as neural
from priority_14235_models import _encode
from train_priority_14235 import sha, write

ROOT = Path(__file__).resolve().parent
RUN = neural.RUN.parent / 'gradients'
EXPECTED_PLAN = '7a94001fe61846f105a460cedb3804673f26a7e8b1bb4ceab13cb13c38e72ee7'


def now():
    return datetime.now(timezone.utc).isoformat()


def freeze():
    RUN.mkdir(parents=True, exist_ok=True)
    path = RUN / 'plan.json'
    assert sha(neural.PLAN) == EXPECTED_PLAN
    plan = json.loads(neural.PLAN.read_text())
    for name, digest in plan['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    if path.exists():
        result = json.loads(path.read_text())
        assert result['source_sha256'] == sha(__file__)
        return result
    result = {'status': 'frozen_before_diagnostic', 'created_utc': now(),
              'source_sha256': sha(__file__), 'neural_plan_sha256': EXPECTED_PLAN,
              'data_sha256': {n: sha(neural.DATA / n) for n in ('fit_x.npy', 'fit_y.npy')},
              'data_scope': 'First64 training objects only, fixed order; no validation, report, extra or confirmation features/labels.',
              'gradient_gate': 'Compare original zero final fusion layer with only that layer set to fixed Gaussian std0.01.',
              'gradient_physical': 'One fixed training input, all540 readout derivatives: PyTorch complex64 autograd vs PennyLane float64 adjoint. Nine prespecified manual parameter-shift checks.',
              'parameter_shift_theta_indices': [[0,0,1], [5,4,0], [11,8,2]],
              'parameter_shift_encoding_indices': [[0,0], [5,4], [11,8]],
              'cpu_benchmark': 'Four threads, batch64, initial parameters; forward warmup3/repeats15 and forward+backward warmup2/repeats5. Mixed server load, descriptive simulator cost only.',
              'barren_plateau_scope': 'A local gradient check cannot establish or exclude gradient variance decay with increasing system size.',
              'training_or_selection_performed': False, 'confirmation_accessed': False}
    write(path, result)
    return result


def stats(values):
    array = values.detach().cpu().numpy().reshape(-1)
    return {'parameters': int(array.size), 'l2_norm': float(np.linalg.norm(array)),
            'rms': float(np.sqrt(np.mean(array*array))), 'max_abs': float(np.abs(array).max()),
            'exact_zero_fraction': float(np.mean(array == 0)), 'finite': bool(np.isfinite(array).all())}


def gradient_gate(x, y, weights):
    rows = []
    models = []
    for label in ('zero_final_head', 'only_final_head_std0p01'):
        model = neural.make_model({'kind': 'quantum', 'seed': 42}).eval()
        if models:
            model.load_state_dict(models[0].state_dict())
        if label != 'zero_final_head':
            generator = torch.Generator().manual_seed(2026101401)
            with torch.no_grad():
                model.fusion_head[-1].weight.copy_(.01*torch.randn(model.fusion_head[-1].weight.shape, generator=generator))
            for key, value in models[0].state_dict().items():
                if key != 'fusion_head.2.weight':
                    assert torch.equal(value, model.state_dict()[key]), key
        logits, aux = model(x, return_aux=True)
        loss = F.cross_entropy(logits, y, weight=weights)
        loss.backward()
        gradients = {name: stats(parameter.grad) for name, parameter in model.named_parameters()
                     if name in ('readout.theta', 'readout.frequency', 'readout.shift', 'fusion_head.0.weight', 'fusion_head.2.weight')}
        assert all(v['finite'] for v in gradients.values())
        if label == 'zero_final_head':
            assert float(aux['residual'].abs().max()) == 0
            assert all(v['max_abs'] == 0 for key,v in gradients.items() if key != 'fusion_head.2.weight')
            assert gradients['fusion_head.2.weight']['max_abs'] > 0
        else:
            assert all(gradients[key]['l2_norm'] > 1e-7 for key in ('readout.theta', 'readout.frequency', 'readout.shift'))
        rows.append({'condition': label, 'loss': float(loss), 'gradients': gradients,
                     'residual_max_abs': float(aux['residual'].abs().max())})
        models.append(model)
    return {'status': 'passed', 'objects': len(x), 'rows': rows,
            'interpretation': 'Zero final residual weights block the initial supervised quantum gradient algebraically. Opening the head restores gradients at these same circuit parameters.'}


def physical_gradient(x, plan):
    import pennylane as qml
    from pennylane import numpy as pnp
    model = neural.make_model({'kind': 'quantum', 'seed': 42}).eval()
    amp, angles = _encode(x[:1], 9, model.feature_permutations[0], model.zero_basis)
    amplitude = amp[0].numpy().astype(float)
    amplitude /= np.linalg.norm(amplitude)
    angle = angles[0].numpy().astype(float)
    readout = model.readout
    coeff = np.linspace(-.7, .9, 27)
    coeff /= np.linalg.norm(coeff)
    raw = readout(amp, angles)
    objective = (raw[0]*torch.tensor(coeff, dtype=torch.float32)).sum()
    actual = torch.autograd.grad(objective, (readout.theta, readout.frequency, readout.shift))
    params = [parameter.detach().numpy().astype(float) for parameter in (readout.theta, readout.frequency, readout.shift)]
    device = qml.device('default.qubit', wires=9)
    operators = [operator(q) for operator in (qml.PauliX, qml.PauliY, qml.PauliZ) for q in range(9)]
    @qml.qnode(device, interface='autograd', diff_method='adjoint')
    def circuit(theta, frequency, shift):
        qml.AmplitudeEmbedding(amplitude, wires=list(reversed(range(9))), normalize=True)
        for layer in range(12):
            for q in range(9):
                qml.RZ(theta[layer,q,0], wires=q)
                qml.RY(theta[layer,q,1], wires=q)
                qml.RZ(theta[layer,q,2], wires=q)
            for q in range(9):
                qml.CNOT(wires=[q,(q+1)%9])
            for q in range(9):
                qml.RZ(angle[q]*frequency[layer,q]+shift[layer,q], wires=q)
        return qml.expval(qml.dot(coeff, operators))
    differentiable = [pnp.array(p, requires_grad=True) for p in params]
    reference = qml.grad(circuit, argnum=(0,1,2))(*differentiable)
    checks = []
    for name, a, b in zip(('theta','frequency','shift'), actual, reference):
        error = float(np.max(np.abs(a.detach().numpy()-np.asarray(b))))
        assert error < 2e-5, (name, error)
        checks.append({'parameter': name, 'max_abs_error': error, 'gradient': stats(a)})
    shifted = []
    for index in plan['parameter_shift_theta_indices']:
        idx = tuple(index)
        plus, minus = params[0].copy(), params[0].copy()
        plus[idx] += np.pi/2; minus[idx] -= np.pi/2
        derivative = .5*(float(circuit(plus,params[1],params[2]))-float(circuit(minus,params[1],params[2])))
        error = abs(derivative-float(reference[0][idx]))
        assert error < 1e-8
        shifted.append({'parameter': 'theta', 'index': index, 'shift_derivative': derivative, 'adjoint_error': error})
    for index in plan['parameter_shift_encoding_indices']:
        idx = tuple(index)
        plus, minus = params[2].copy(), params[2].copy()
        plus[idx] += np.pi/2; minus[idx] -= np.pi/2
        gate_derivative = .5*(float(circuit(params[0],params[1],plus))-float(circuit(params[0],params[1],minus)))
        for name, group, derivative in (('shift',2,gate_derivative), ('frequency',1,angle[idx[1]]*gate_derivative)):
            error = abs(derivative-float(reference[group][idx]))
            assert error < 1e-8
            shifted.append({'parameter': name, 'index': index, 'shift_derivative': derivative, 'adjoint_error': error})
    forward_error = abs(float(objective)-float(circuit(*params)))
    assert forward_error < 5e-6
    return {'status': 'passed', 'objects': 1, 'readout_parameters_checked': 540,
            'all_gradient_checks': checks, 'manual_shift_checks': shifted, 'forward_max_error': forward_error,
            'versions': {'torch': torch.__version__, 'pennylane': qml.__version__},
            'scope': 'One deterministic training input and scalar linear readout probe; verifies derivatives, not generalization or absence of barren plateaus.'}


def benchmark(x, y, weights):
    rows = []
    for kind in ('quantum','matched_mlp','pure_mlp','strong_mlp'):
        model = neural.make_model({'kind':kind,'seed':42}).eval()
        with torch.no_grad():
            for _ in range(3):model(x)
            forward = []
            for _ in range(15):
                start = time.perf_counter();model(x);forward.append(time.perf_counter()-start)
        backward = []
        for repeat in range(7):
            start = time.perf_counter()
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x),y,weight=weights).backward()
            if repeat >= 2:backward.append(time.perf_counter()-start)
        rows.append({'kind':kind,'parameters':model.counts()['total'],'batch_size':len(x),
                     'forward_seconds':forward,'forward_median_seconds':float(np.median(forward)),
                     'forward_backward_seconds':backward,'forward_backward_median_seconds':float(np.median(backward))})
    return {'status':'complete','device':'CPU','torch_threads':torch.get_num_threads(),'rows':rows,
            'scope':'Initial-weight architecture timing on shared CPU under concurrent training; includes encoding/backbone/fusion, excludes optimizer and quantum hardware.'}


def execute(plan):
    state={'status':'running','pid':os.getpid(),'started_utc':now(),'plan_sha256':sha(RUN/'plan.json')}
    write(RUN/'state.json',state)
    try:
        for name,digest in plan['data_sha256'].items():assert sha(neural.DATA/name)==digest,name
        x=torch.from_numpy(np.load(neural.DATA/'fit_x.npy',mmap_mode='r')[:64].copy())
        ycpu=np.load(neural.DATA/'fit_y.npy');y=torch.from_numpy(ycpu[:64].copy())
        freq=np.bincount(ycpu,minlength=10)
        weights=torch.tensor(len(ycpu)/(10*freq),dtype=torch.float32)
        gate=gradient_gate(x,y,weights);write(RUN/'gradient_gate.json',gate)
        state.update(stage='physical_gradient',checked_utc=now());write(RUN/'state.json',state)
        physical=physical_gradient(x,plan);write(RUN/'physical_gradient.json',physical)
        state.update(stage='cpu_benchmark',checked_utc=now());write(RUN/'state.json',state)
        cost=benchmark(x,y,weights);write(RUN/'cpu_cost.json',cost)
        result={'status':'passed','plan_sha256':sha(RUN/'plan.json'),'gradient_gate':gate,
                'physical_gradient':physical,'cpu_cost':cost,'platform':platform.platform(),
                'training_or_selection_performed':False,'confirmation_accessed':False,'finished_utc':now()}
        write(RUN/'result.json',result)
        state.update(status='complete',checked_utc=now());write(RUN/'state.json',state)
    except Exception as error:
        state.update(status='failed',error=repr(error),traceback=traceback.format_exc(),checked_utc=now())
        write(RUN/'state.json',state);raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','launch','run','status'))
    args=parser.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    plan=freeze()
    if args.command=='freeze':out={'status':'frozen','plan_sha256':sha(RUN/'plan.json')}
    elif args.command=='run':execute(plan);out={'status':'complete'}
    elif args.command=='status':
        state=json.loads((RUN/'state.json').read_text()) if (RUN/'state.json').exists() else {}
        proc=Path('/proc')/str(state.get('pid'))/'cmdline'
        out={'state':state,'alive':proc.exists() and Path(__file__).name in proc.read_bytes().decode(),
             'result':json.loads((RUN/'result.json').read_text()) if (RUN/'result.json').exists() else None}
    else:
        receipt=json.loads((RUN/'launch.json').read_text()) if (RUN/'launch.json').exists() else {}
        proc=Path('/proc')/str(receipt.get('pid'))/'cmdline'
        if proc.exists() and Path(__file__).name in proc.read_bytes().decode():out=dict(receipt,already_running=True)
        elif (RUN/'result.json').exists():out={'status':'complete'}
        else:
            with (RUN/'run.log').open('ab',buffering=0) as log:
                process=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'run'],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
            out={'status':'launched','pid':process.pid,'utc':now(),'plan_sha256':sha(RUN/'plan.json')};write(RUN/'launch.json',out)
    print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
