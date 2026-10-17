"""Recompute frozen priority-study predictions and both independent selections."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from priority_14235_models import make_model
from train_priority_14235 import DATA, PLAN, RUN, alpha_grid, metric, predict, sha, validate_plan, write


def selected(history, metric_name):
    other = 'accuracy' if metric_name == 'macro_f1' else 'macro_f1'
    return max(
        (dict(value, epoch=row['epoch']) for row in history for value in row['validation_grid']),
        key=lambda row: (row['metrics'][metric_name], row['metrics'][other]),
    )


def audit(name):
    plan = json.loads(PLAN.read_text())
    validate_plan(plan)
    job = next(job for job in plan['jobs'] if job['name'] == name)
    folder = RUN / 'models' / name
    result = json.loads((folder / 'training_result.json').read_text())
    accuracy_selection = json.loads((folder / 'accuracy_selection.json').read_text())
    assert result['plan_sha256'] == sha(PLAN)
    assert accuracy_selection['plan_sha256'] == sha(PLAN)
    history = result['history']
    assert [row['epoch'] for row in history] == list(range(job['epochs'] + 1))
    macro = selected(history, 'macro_f1')
    acc = selected(history, 'accuracy')
    assert macro['epoch'] == result['best_macro']['epoch']
    assert macro['alpha'] == result['best_macro']['alpha']
    assert acc == accuracy_selection['candidate']
    assert sha(folder / 'best_accuracy.pt') == accuracy_selection['checkpoint_sha256']
    assert sha(folder / 'best_macro.pt') == result['checkpoint_sha256']['macro']
    latest = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
    assert latest['epoch'] == job['epochs'] and latest['cursor'] == 0
    expected_steps = job['epochs'] * int(np.ceil(633034 / plan['batch_size']))
    steps = {int(value['step'].item()) for value in latest['optimizer']['state'].values()}
    assert steps == {expected_steps}, (steps, expected_steps)
    model = make_model(job['spec']).eval()
    assert model.counts() == result['counts']
    frequency = np.bincount(np.load(DATA / 'fit_y.npy'), minlength=10)
    logweights = np.log(frequency.sum() / (10 * frequency))
    complete = {}
    recomputations = {}
    for label, checkpoint, candidate in (
        ('best_macro', folder / 'best_macro.pt', macro),
        ('best_accuracy', folder / 'best_accuracy.pt', acc),
    ):
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        metrics = {}
        checks = {}
        for split, prefix in (('validation', 'val'), ('report', 'report'), ('extra', 'extra')):
            x = torch.from_numpy(np.load(DATA / (prefix + '_x.npy')))
            y = np.load(DATA / (prefix + '_y.npy'))
            ids = np.load(DATA / (prefix + '_ids.npy'))
            logits = predict(model, x, batch=256)
            corrected = logits - candidate['alpha'] * logweights
            pred = corrected.argmax(1)
            metrics[split] = metric(y, pred)
            if split == 'validation':
                _, grid_macro = alpha_grid(y, logits, logweights)
                expected = next(row for row in history if row['epoch'] == candidate['epoch'])
                checkgrid = [metric(y, (logits - value['alpha'] * logweights).argmax(1))
                             for value in expected['validation_grid']]
                assert checkgrid == [value['metrics'] for value in expected['validation_grid']]
                assert metrics[split] == candidate['metrics']
            if label == 'best_macro':
                cached = np.load(folder / (split + '_best_macro_predictions.npz'))
                assert np.array_equal(ids, cached['ids'])
                assert np.array_equal(y, cached['y'])
                disagreements = int(np.count_nonzero(pred != cached['pred']))
                max_error = float(np.max(np.abs(logits - cached['logits'])))
                assert disagreements == 0, (name, split, disagreements)
                assert metrics[split] == result['best_macro']['metrics'][split]
                checks[split] = {'prediction_disagreements': disagreements, 'max_logit_error': max_error,
                                 'objects': len(y)}
            else:
                np.savez_compressed(folder / (split + '_best_accuracy_predictions.npz'),
                                    ids=ids, y=y, pred=pred, logits=logits, corrected_logits=corrected,
                                    prior_alpha=np.asarray(candidate['alpha']))
                checks[split] = {'objects': len(y), 'selected_before_report_extra': True}
        complete[label] = {'epoch': candidate['epoch'], 'alpha': candidate['alpha'], 'metrics': metrics,
                           'checkpoint_sha256': sha(checkpoint)}
        recomputations[label] = checks
    payload = {'status': 'passed', 'name': name, 'priority_direction': job['priority_direction'],
               'counts': result['counts'], 'completed_epochs': job['epochs'], 'optimizer_steps': expected_steps,
               'best_macro': complete['best_macro'], 'best_acc': complete['best_accuracy'],
               'original_macro_conditioned_acc': result['best_acc'],
               'recomputation': recomputations, 'training_result_sha256': sha(folder / 'training_result.json'),
               'accuracy_selection_sha256': sha(folder / 'accuracy_selection.json'),
               'plan_sha256': sha(PLAN), 'confirmation_accessed': False}
    write(folder / 'result_audit.json', payload)
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    try:
        outcome = audit(args.name)
        print(json.dumps({'name': args.name, 'status': 'passed', 'best_macro': outcome['best_macro'],
                          'best_acc': outcome['best_acc']}), flush=True)
    except Exception as error:
        import traceback
        write(RUN / 'models' / args.name / 'result_audit_failure.json',
              {'name': args.name, 'error': repr(error), 'traceback': traceback.format_exc()})
        raise
