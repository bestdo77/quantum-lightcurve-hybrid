"""Matched classical branches for the frozen q9d12 branch-first experiment.

The original training sources and original quantum results are never edited.
Only model construction, parameter grouping, and exact ACC checkpoint capture
are adapted; the original optimizer, loss, ordering, schedule, and training loop
are reused. Both classical branches have 540 meaningful learned parameters.
"""
import argparse
import copy
import json
import math
import os
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import train_priority_14235 as trainer
import audit_priority_14235_results as auditor
from priority_14235_models import PriorityQuantumModel, _encode, make_model as quantum_factory
from safe_capturable_adamw import SafeCapturableAdamW

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'p1_matched_classical_20261014'
PLAN = RUN / 'plan.json'
REFERENCE = ROOT / 'priority_14235_20261013'
QNAME = 'p1_adamw_q9d12_branchfirst'
QDIR = REFERENCE / 'models' / QNAME
OLD_CLASSICAL = REFERENCE / 'models' / 'p4_classical_residual_matched' / 'initial.pt'
GPU = 'GPU-659579c9-052a-ec0f-637c-de675325d6d8'
ORIGINAL_ALPHA = trainer.alpha_grid
ORIGINAL_LOAD_SPLIT = trainer.load_split
sha, write, atomic_torch = trainer.sha, trainer.write, trainer.atomic_torch


def now():
    return datetime.now(timezone.utc).isoformat()


class SparseClassicalReadout(nn.Module):
    """Original 27x20 sparse tanh readout, with matched optimizer allocation."""
    def __init__(self):
        super().__init__()
        state = torch.load(OLD_CLASSICAL, map_location='cpu', weights_only=True)
        weight = state['readout.weight']
        self.fast_weight = nn.Parameter(weight[:, :12].clone())
        self.slow_weight = nn.Parameter(weight[:, 12:].clone())
        self.register_buffer('indices', state['readout.indices'].clone())

    def fast_parameters(self):
        return [self.fast_weight]

    def slow_parameters(self):
        return [self.slow_weight]

    def forward(self, x):
        selected = torch.atan(x)[:, self.indices]
        weight = torch.cat((self.fast_weight, self.slow_weight), 1)
        return torch.tanh((selected * weight.unsqueeze(0)).sum(-1) / math.sqrt(20.0))


class ProjectedMLPReadout(nn.Module):
    """Fixed full-feature projection -> ordinary 26->10->27 GELU MLP.

    Learned counts: first linear 260+10; second bias-free linear 270.
    270 first-layer parameters + 54 output weights receive the fast rate;
    the remaining 216 output weights receive the slow, zero-decay rate.
    The allocation is frozen before observing any new validation result.
    """
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261014)
            self.first = nn.Linear(26, 10)
            head = nn.Linear(10, 27, bias=False)
        self.fast_head = nn.Parameter(head.weight[:, :2].detach().clone())
        self.slow_head = nn.Parameter(head.weight[:, 2:].detach().clone())
        generator = torch.Generator().manual_seed(20261014)
        projection = 2 * torch.randint(0, 2, (359 + 9, 26), generator=generator).float() - 1
        self.register_buffer('projection', projection)
        self.register_buffer('permutation', torch.arange(359))
        basis = torch.zeros(512)
        basis[0] = 1
        self.register_buffer('zero_basis', basis)

    def fast_parameters(self):
        return list(self.first.parameters()) + [self.fast_head]

    def slow_parameters(self):
        return [self.slow_head]

    def forward(self, x):
        amplitude, angles = _encode(x, 9, self.permutation, self.zero_basis)
        projected = torch.cat((amplitude[:, :359], angles), -1) @ self.projection
        hidden = F.gelu(self.first(projected))
        return F.linear(hidden, torch.cat((self.fast_head, self.slow_head), 1))


class ClassicalControl(PriorityQuantumModel):
    def __init__(self, kind):
        super().__init__('classical_residual_control', qubits=9, depth=12, registers=1, fusion_hidden=128)
        self.kind = kind
        self.readout = SparseClassicalReadout() if kind == 'sparse' else ProjectedMLPReadout()
        # Shared M8, normalization, and fusion tensors match the quantum initial
        # checkpoint exactly, rather than merely sharing a random seed.
        source = torch.load(QDIR / 'initial.pt', map_location='cpu', weights_only=True)
        state = self.state_dict()
        for key, value in state.items():
            if not key.startswith('readout.'):
                assert key in source and value.shape == source[key].shape
                value.copy_(source[key])
        self.load_state_dict(state)

    def counts(self):
        counts = super().counts()
        counts.update(classical_readout=self.kind,
                      readout_parameters=sum(p.numel() for p in self.readout.parameters()),
                      fast_readout_parameters=sum(p.numel() for p in self.readout.fast_parameters()),
                      slow_readout_parameters=sum(p.numel() for p in self.readout.slow_parameters()))
        assert counts['total'] == 44238 and counts['quantum'] == counts['quantum_encoding'] == 0
        assert counts['readout_parameters'] == 540
        assert counts['fast_readout_parameters'] == 324 and counts['slow_readout_parameters'] == 216
        return counts


def make_model(spec):
    return ClassicalControl(spec['kind'])


def optimizer(model, job, device):
    readout_ids = {id(p) for p in model.readout.parameters()}
    backbone = list(model.backbone.parameters())
    branch = [p for name, p in model.named_parameters()
              if not name.startswith('backbone.') and id(p) not in readout_ids]
    buckets = [
        ('backbone', backbone, 5e-5, 1e-5, 1e-4, 10),
        ('theta', model.readout.fast_parameters(), 8e-4, 8e-5, 1e-5, 0),
        ('encoding', model.readout.slow_parameters(), 5e-4, 5e-5, 0., 0),
        ('branch', branch, 1e-3, 1e-4, 1e-4, 0),
    ]
    groups = [{'params': params, 'lr': torch.tensor(0. if warm else start, device=device),
               'start': start, 'end': end, 'weight_decay': decay, 'warmup_epochs': warm,
               'group_name': name}
              for name, params, start, end, decay, warm in buckets]
    assert [sum(p.numel() for p in group['params']) for group in groups] == [23170, 324, 216, 20528]
    assert len({id(p) for group in groups for p in group['params']}) == len(list(model.parameters()))
    opt = SafeCapturableAdamW(groups, eps=1e-8, capturable=device.type == 'cuda', foreach=False)
    for p in model.parameters():
        opt.state[p] = {'step': torch.zeros((), device=device), 'exp_avg': torch.zeros_like(p),
                        'exp_avg_sq': torch.zeros_like(p)}
    return opt


def gradients(model):
    return {name: {'rms': float(p.grad.detach().square().mean().sqrt()),
                   'max_abs': float(p.grad.detach().abs().max())}
            for name, p in model.named_parameters() if name.startswith('readout.') and p.grad is not None}


def configure():
    trainer.RUN, trainer.PLAN = RUN, PLAN
    trainer.make_model, trainer.optimizer, trainer.gradient_stats = make_model, optimizer, gradients
    auditor.RUN, auditor.PLAN, auditor.make_model = RUN, PLAN, make_model


def validate(plan):
    trainer.validate_plan(plan)
    assert sha(REFERENCE / 'plan.json') == plan['reference_plan_sha256']
    assert sha(QDIR / 'initial.pt') == plan['quantum_initial_sha256']
    assert sha(QDIR / 'result_audit.json') == plan['quantum_audit_sha256']
    assert sha(OLD_CLASSICAL) == plan['old_classical_initial_sha256']


def prepare():
    RUN.mkdir(exist_ok=True)
    if PLAN.exists():
        plan = json.loads(PLAN.read_text())
        validate(plan)
        return plan
    original = json.loads((REFERENCE / 'plan.json').read_text())
    qjob = next(job for job in original['jobs'] if job['name'] == QNAME)
    assert json.loads((QDIR / 'result_audit.json').read_text())['status'] == 'passed'
    jobs = []
    for kind in ('sparse', 'mlp'):
        job = copy.deepcopy(qjob)
        job.update(name='p1_classical_' + kind + '_matched', role='matched_classical_control',
                   spec={'kind': kind}, freeze_backbone=False)
        assert job['objective'] == 'standard' and job['backbone_warmup_epochs'] == 10 and job['epochs'] == 40
        jobs.append(job)
    plan = copy.deepcopy(original)
    plan.update(created_utc=now(), jobs=jobs, main_result_models=[job['name'] for job in jobs],
                comparison='Two 44238-parameter classical branches versus the fixed p1 q9d12 model.',
                reference_plan_sha256=sha(REFERENCE / 'plan.json'),
                quantum_initial_sha256=sha(QDIR / 'initial.pt'), quantum_audit_sha256=sha(QDIR / 'result_audit.json'),
                old_classical_initial_sha256=sha(OLD_CLASSICAL),
                classical_branches={
                    'sparse': 'Original fixed 27x20 sparse tanh readout, split 324/216 for identical optimizer-group budgets.',
                    'mlp': 'Fixed seed20261014 dense sign projection of normalized atan amplitudes and the same nine angles to 26 coordinates; learned 26->10->27 GELU MLP, 540 parameters. No fitted projection or padding parameters.'},
                limitations='One seed; original quantum result already observed. Tests these specified classical branches, not all possible equal-parameter MLPs. Report/extra are exposed development cohorts.',
                selection='Independent maxima over epoch x alpha grid for Macro-F1 and ACC. Capture every epoch weights and freeze both before report/extra.',
                authorization='User requested a same-parameter, same-training-protocol comparison for direction 1.',
                counts={job['name']: make_model(job['spec']).counts() for job in jobs})
    plan['source_sha256']['p1_matched_classical_comparison.py'] = sha(Path(__file__))
    write(PLAN, plan)
    validate(plan)
    return plan


def preflight(plan):
    validate(plan)
    x = torch.from_numpy(np.load(trainer.DATA / 'val_x.npy', mmap_mode='r')[:64].copy())
    y = torch.from_numpy(np.load(trainer.DATA / 'val_y.npy', mmap_mode='r')[:64].copy())
    source = torch.load(QDIR / 'initial.pt', map_location='cpu', weights_only=True)
    qjob = next(job for job in json.loads((REFERENCE / 'plan.json').read_text())['jobs'] if job['name'] == QNAME)
    quantum = quantum_factory(qjob['spec']).eval()
    quantum.load_state_dict(source)
    with torch.no_grad():
        reference_logits = quantum(x)
    rows = []
    for job in plan['jobs']:
        cpu = make_model(job['spec']).eval()
        shared = [key for key in cpu.state_dict() if not key.startswith('readout.')]
        assert all(torch.equal(cpu.state_dict()[key], source[key]) for key in shared)
        with torch.no_grad():
            initial_error = float((cpu(x) - reference_logits).abs().max())
        assert initial_error == 0
        if job['spec']['kind'] == 'mlp':
            assert torch.linalg.matrix_rank(cpu.readout.projection) == 26
            assert (cpu.readout.projection.abs().sum(1) > 0).all()
        eager = make_model(job['spec']).cuda().train()
        graphed = make_model(job['spec']).cuda().train()
        init = {key: value.detach().clone() for key, value in eager.state_dict().items()}
        xe, ye, weights = x.cuda(), y.cuda(), torch.ones(10, device='cuda')
        oe, og = optimizer(eager, job, torch.device('cuda')), optimizer(graphed, job, torch.device('cuda'))
        graph = trainer.GraphStep(graphed, og, xe, ye, weights, job)
        for _ in range(3):
            le = trainer.eager_step(eager, oe, xe, ye, weights, job)
            lg = graph(xe, ye)
            assert math.isfinite(float(le)) and math.isfinite(float(lg))
        torch.cuda.synchronize()
        error = max(float((value - graphed.state_dict()[key]).abs().max())
                    for key, value in eager.state_dict().items() if value.is_floating_point())
        assert error < 3e-5, error
        assert all(torch.equal(init[key], eager.state_dict()[key]) for key in init if key.startswith('backbone.'))
        updates = {name: float((p.detach() - init[name]).abs().max())
                   for name, p in eager.named_parameters() if name.startswith('readout.')}
        assert updates and all(value > 0 for value in updates.values()), updates
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in eager.parameters())
        rows.append({'name': job['name'], 'counts': cpu.counts(), 'shared_initial_tensors_equal': len(shared),
                     'initial_logit_error_vs_quantum': initial_error, 'eager_graph_max_error': error,
                     'readout_parameter_updates': updates, 'warmup_backbone_unchanged': True,
                     'optimizer_group_parameters': [sum(p.numel() for p in group['params']) for group in oe.param_groups]})
        del cpu, eager, graphed, oe, og, graph
        torch.cuda.empty_cache()
    result = {'status': 'passed', 'rows': rows, 'plan_sha256': sha(PLAN), 'confirmation_accessed': False}
    write(RUN / 'preflight.json', result)
    return result


def capture_validation_checkpoints(job, plan):
    folder = RUN / 'models' / job['name']
    latest = folder / 'latest.pt'
    next_epoch = torch.load(latest, map_location='cpu', weights_only=False)['epoch'] + 1 if latest.exists() else 0
    runtime = {'model': None, 'epoch': next_epoch}

    def factory(spec):
        model = make_model(spec)
        runtime['model'] = model
        return model

    def grid(y, logits, logweights):
        values, macro = ORIGINAL_ALPHA(y, logits, logweights)
        assert runtime['model'] is not None
        checkpoint = folder / 'epoch_checkpoints' / ('epoch_%03d.pt' % runtime['epoch'])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch(runtime['model'].state_dict(), checkpoint)
        runtime['epoch'] += 1
        return values, macro

    def freeze_before_holdouts(split, device):
        assert split in ('report', 'extra')
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        assert saved['epoch'] == job['epochs'] and saved['cursor'] == 0
        candidate = auditor.selected(saved['history'], 'accuracy')
        source = folder / 'epoch_checkpoints' / ('epoch_%03d.pt' % candidate['epoch'])
        destination = folder / 'best_accuracy.pt'
        state = torch.load(source, map_location='cpu', weights_only=True)
        atomic_torch(state, destination)
        write(folder / 'accuracy_selection.json', {'status': 'frozen_validation_selection', 'candidate': candidate,
              'source': str(source), 'selected_utc': now(), 'checkpoint_sha256': sha(destination),
              'plan_sha256': sha(PLAN), 'report_extra_used_in_selection': False,
              'selection': 'ACC first over all epoch x alpha candidates; Macro-F1 breaks ties.',
              'confirmation_accessed': False})
        return ORIGINAL_LOAD_SPLIT(split, device)

    trainer.make_model, trainer.alpha_grid, trainer.load_split = factory, grid, freeze_before_holdouts


def protocol_audit(job, plan, audit):
    folder = RUN / 'models' / job['name']
    initial = torch.load(folder / 'initial.pt', map_location='cpu', weights_only=True)
    reference_initial = torch.load(QDIR / 'initial.pt', map_location='cpu', weights_only=True)
    latest = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
    original_history = json.loads((QDIR / 'training_result.json').read_text())['history']
    for key in initial:
        if not key.startswith('readout.'):
            assert torch.equal(initial[key], reference_initial[key]), key
    assert [row['learning_rates'] for row in latest['history'][1:]] == [row['learning_rates'] for row in original_history[1:]]
    expected_parameters = [23170, 324, 216, 20528]
    group_parameters = [sum(latest['optimizer']['state'][index]['exp_avg'].numel() for index in group['params'])
                        for group in latest['optimizer']['param_groups']]
    assert group_parameters == expected_parameters
    reference_latest = torch.load(QDIR / 'latest.pt', map_location='cpu', weights_only=False)
    for actual, expected in zip(latest['optimizer']['param_groups'], reference_latest['optimizer']['param_groups']):
        for key in ('group_name', 'start', 'end', 'weight_decay', 'warmup_epochs', 'betas', 'eps', 'capturable', 'foreach'):
            assert actual[key] == expected[key], (key, actual[key], expected[key])
    assert latest['epoch'] == job['epochs'] == 40 and latest['cursor'] == 0
    for row in latest['history'][1:]:
        assert (row['learning_rates']['backbone'] == 0) == (row['epoch'] <= 10)
    epoch10 = torch.load(folder / 'epoch_checkpoints' / 'epoch_010.pt', map_location='cpu', weights_only=True)
    assert all(torch.equal(initial[key], epoch10[key]) for key in initial if key.startswith('backbone.'))
    assert any(not torch.equal(initial[key], latest['model'][key]) for key in initial if key.startswith('backbone.'))
    for epoch in range(41):
        assert (folder / 'epoch_checkpoints' / ('epoch_%03d.pt' % epoch)).is_file()
    result = {'status': 'passed', 'name': job['name'], 'total_parameters': 44238,
              'readout_parameters': 540, 'shared_initial_tensors_exactly_equal': True,
              'full_learning_rate_history_matches_quantum': True, 'optimizer_group_parameters': group_parameters,
              'first_ten_epochs_backbone_unchanged': True, 'later_backbone_updated': True,
              'optimizer_steps': audit['optimizer_steps'], 'epochs': 40,
              'objective': job['objective'], 'result_audit_sha256': sha(folder / 'result_audit.json'),
              'plan_sha256': sha(PLAN), 'confirmation_accessed': False}
    assert result['optimizer_steps'] == 98920
    write(folder / 'protocol_audit.json', result)
    return result


def aggregate(plan):
    quantum = json.loads((QDIR / 'result_audit.json').read_text())
    controls, pairs = [], []
    for job in plan['jobs']:
        folder = RUN / 'models' / job['name']
        audit = json.loads((folder / 'result_audit.json').read_text())
        protocol = json.loads((folder / 'protocol_audit.json').read_text())
        assert audit['status'] == protocol['status'] == 'passed'
        controls.append(audit)
        for label, prefix in (('best_macro', 'best_macro'), ('best_acc', 'best_accuracy')):
            for split in ('validation', 'report', 'extra'):
                qpred = np.load(QDIR / (split + '_' + prefix + '_predictions.npz'))
                cpred = np.load(folder / (split + '_' + prefix + '_predictions.npz'))
                assert np.array_equal(qpred['ids'], cpred['ids']) and np.array_equal(qpred['y'], cpred['y'])
                qc, cc = qpred['pred'] == qpred['y'], cpred['pred'] == cpred['y']
                pairs.append({'name': job['name'], 'selection': label, 'split': split,
                    'quantum_only_correct': int(np.count_nonzero(qc & ~cc)),
                    'classical_only_correct': int(np.count_nonzero(~qc & cc)),
                    'prediction_disagreements': int(np.count_nonzero(qpred['pred'] != cpred['pred'])),
                    'quantum_minus_classical': {key: quantum[label]['metrics'][split][key] - audit[label]['metrics'][split][key]
                                               for key in ('macro_f1', 'accuracy')}})
    result = {'status': 'complete', 'quantum': quantum, 'controls': controls, 'paired_comparisons': pairs,
              'plan_sha256': sha(PLAN), 'reference_plan_sha256': plan['reference_plan_sha256'],
              'limitations': plan['limitations'], 'confirmation_accessed': False}
    write(RUN / 'comparison_summary.json', result)
    return result


def execute(plan):
    validate(plan)
    check = json.loads((RUN / 'preflight.json').read_text())
    assert check['status'] == 'passed' and check['plan_sha256'] == sha(PLAN)
    state = {'status': 'training', 'pid': os.getpid(), 'started_utc': now(), 'completed': [], 'plan_sha256': sha(PLAN)}
    write(RUN / 'state.json', state)
    try:
        for job in plan['jobs']:
            state.update(current=job['name'], checked_utc=now())
            write(RUN / 'state.json', state)
            capture_validation_checkpoints(job, plan)
            trainer.train(job, plan, torch.device('cuda'))
            trainer.make_model, trainer.alpha_grid, trainer.load_split = make_model, ORIGINAL_ALPHA, ORIGINAL_LOAD_SPLIT
            torch.cuda.empty_cache()
            state.update(status='auditing', checked_utc=now())
            write(RUN / 'state.json', state)
            audit = auditor.audit(job['name'])
            protocol_audit(job, plan, audit)
            state['completed'].append(job['name'])
            state.update(status='training', checked_utc=now())
            write(RUN / 'state.json', state)
        aggregate(plan)
        state.update(status='complete', current=None, finished_utc=now(), checked_utc=now())
        write(RUN / 'state.json', state)
    except Exception as error:
        state.update(status='failed', error=repr(error), traceback=traceback.format_exc(), checked_utc=now())
        write(RUN / 'state.json', state)
        raise


def launch(plan):
    previous = json.loads((RUN / 'state.json').read_text()) if (RUN / 'state.json').exists() else {}
    receipt = json.loads((RUN / 'launcher_receipt.json').read_text()) if (RUN / 'launcher_receipt.json').exists() else {}
    pid = previous.get('pid') or receipt.get('pid')
    if previous.get('status') == 'complete':
        return {'status': 'complete', 'already_complete': True}
    if pid:
        cmdline = Path('/proc') / str(pid) / 'cmdline'
        if cmdline.exists() and 'p1_matched_classical_comparison.py' in cmdline.read_bytes().decode():
            return {'status': 'running', 'pid': pid, 'already_running': True}
    assert json.loads((RUN / 'preflight.json').read_text())['status'] == 'passed'
    with (RUN / 'training.log').open('ab', buffering=0) as log:
        process = subprocess.Popen([str(ROOT / 'qml-env/bin/python'), '-u', str(Path(__file__)), 'run'],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=GPU,
                OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
    receipt = {'status': 'launched', 'pid': process.pid, 'started_utc': now(), 'plan_sha256': sha(PLAN), 'gpu_uuid': GPU}
    write(RUN / 'launcher_receipt.json', receipt)
    return receipt


def status(plan):
    def read(path):
        return json.loads(path.read_text()) if path.exists() else None
    return {'utc': now(), 'plan_sha256': sha(PLAN), 'state': read(RUN / 'state.json'),
            'launch': read(RUN / 'launcher_receipt.json'), 'preflight': read(RUN / 'preflight.json'),
            'models': [{'name': job['name'], 'progress': read(RUN / 'models' / job['name'] / 'progress.json'),
                        'audit': read(RUN / 'models' / job['name'] / 'result_audit.json'),
                        'protocol': read(RUN / 'models' / job['name'] / 'protocol_audit.json')}
                       for job in plan['jobs']], 'comparison': read(RUN / 'comparison_summary.json')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'check', 'launch', 'run', 'status'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    configure()
    plan = prepare()
    if args.command == 'prepare':
        result = {'status': 'frozen', 'plan_sha256': sha(PLAN), 'counts': plan['counts']}
    elif args.command == 'check':
        assert torch.cuda.is_available() and 'L20' in torch.cuda.get_device_name(0)
        result = preflight(plan)
    elif args.command == 'launch':
        result = launch(plan)
    elif args.command == 'run':
        assert torch.cuda.is_available() and 'L20' in torch.cuda.get_device_name(0)
        execute(plan)
        result = {'status': 'complete'}
    else:
        result = status(plan)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
