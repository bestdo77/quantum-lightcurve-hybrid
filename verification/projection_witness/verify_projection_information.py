"""Reproduce a branch-level representation distinction on synthetic inputs.

No data access or training is performed. The parameter JSON contains only the
two readouts and the fixed projection, not the pretrained classical backbone.
This is not a classification benchmark or a quantum computational advantage.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from quantum_readout import QuantumReadout, encode


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group_map():
    matrix = np.zeros((9,359),dtype=np.float64)
    for q in range(9):
        matrix[q,q*40:min((q+1)*40,359)] = 2./40
    return matrix


def classical_readout(parameters, amplitude, angles):
    projection = torch.tensor(parameters['projection'],dtype=amplitude.dtype)
    weight = torch.tensor(parameters['first_weight'],dtype=amplitude.dtype)
    bias = torch.tensor(parameters['first_bias'],dtype=amplitude.dtype)
    head = torch.tensor(parameters['head_weight'],dtype=amplitude.dtype)
    z = torch.cat((amplitude[:,:359],angles),-1) @ projection
    return F.linear(F.gelu(F.linear(z,weight,bias)),head)


def physical_reference(readout,amplitude,angles):
    import pennylane as qml
    dev = qml.device('default.qubit',wires=9)
    theta = readout.theta.detach().numpy().astype(float)
    frequency = readout.frequency.detach().numpy().astype(float)
    shift = readout.shift.detach().numpy().astype(float)
    @qml.qnode(dev,diff_method=None)
    def circuit(psi,a):
        qml.AmplitudeEmbedding(psi,wires=list(reversed(range(9))),normalize=True)
        for layer in range(12):
            for q in range(9):
                qml.RZ(theta[layer,q,0],wires=q)
                qml.RY(theta[layer,q,1],wires=q)
                qml.RZ(theta[layer,q,2],wires=q)
            for q in range(9): qml.CNOT(wires=[q,(q+1)%9])
            for q in range(9): qml.RZ(a[q]*frequency[layer,q]+shift[layer,q],wires=q)
        return tuple(qml.expval(op(q)) for op in (qml.PauliX,qml.PauliY,qml.PauliZ) for q in range(9))
    references=[]
    for psi,a in zip(amplitude.numpy().astype(float),angles.numpy().astype(float)):
        references.append(np.asarray(circuit(psi/np.linalg.norm(psi),a),dtype=float))
    return np.stack(references),qml.__version__


def execute(parameters,plan):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    rng=np.random.default_rng(plan['seed'])
    projection=np.asarray(parameters['classical']['projection'],dtype=np.float64)
    assert projection.shape == (368,26)
    amplitude_projection=projection[:359]
    angle_map=group_map()
    base=rng.normal(0,.2,359)
    constraints=np.concatenate((amplitude_projection.T,angle_map,base[None]),axis=0)
    _,singular,vh=np.linalg.svd(constraints,full_matrices=False)
    rank=int(np.count_nonzero(singular>1e-10))
    assert rank <= 36
    basis=vh[:rank]
    raw_pairs=[]
    identities=[]
    for pair in range(plan['pairs']):
        direction=rng.normal(size=359)
        direction-=basis.T@(basis@direction)
        direction*=plan['direction_l2']/np.linalg.norm(direction)
        r=np.stack((base+direction,base-direction))
        norms=np.linalg.norm(r,axis=1)
        angles=r@angle_map.T
        z=(r/norms[:,None])@amplitude_projection+angles@projection[359:]
        errors={'constraint_max_error':float(np.abs(constraints@direction).max()),
                'norm_difference':float(abs(norms[0]-norms[1])),
                'group_angle_max_difference':float(np.abs(angles[0]-angles[1]).max()),
                'projection_max_difference_float64':float(np.abs(z[0]-z[1]).max())}
        assert max(errors.values()) < plan['identity_tolerance']
        assert np.abs(np.tan(r)).max() < 10
        raw_pairs.append(r)
        identities.append(errors)
    raw_pairs=np.stack(raw_pairs)
    inputs=torch.tensor(np.tan(raw_pairs.reshape(-1,359)),dtype=torch.float32)
    amplitude,angles=encode(inputs)
    readout=QuantumReadout(depth=12,cnot=True)
    readout.load_parameters(parameters['quantum'])
    with torch.no_grad():
        quantum=readout(amplitude,angles).numpy().reshape(plan['pairs'],2,27)
        classical=classical_readout(parameters['classical'],amplitude,angles).numpy().reshape(plan['pairs'],2,27)
    rows=[]
    for pair in range(plan['pairs']):
        qdiff=float(np.abs(quantum[pair,0]-quantum[pair,1]).max())
        cdiff=float(np.abs(classical[pair,0]-classical[pair,1]).max())
        assert cdiff < plan['classical_FP32_tolerance']
        rows.append({'pair':pair,**identities[pair],
                     'classical_readout_max_difference_FP32':cdiff,
                     'quantum_readout_max_difference_FP32':qdiff})
    reference,version=physical_reference(readout,amplitude[:2],angles[:2])
    error=float(np.abs(reference-quantum[0]).max())
    assert error < plan['physical_reference_tolerance']
    separated=sum(row['quantum_readout_max_difference_FP32']>plan['separation_threshold'] for row in rows)
    return {'status':'passed','scope':plan['scope'],'pairs':plan['pairs'],'constraint_rank':rank,
            'nullspace_dimension':359-rank,'rows':rows,
            'quantum_pairs_above_threshold':separated,
            'quantum_separation_threshold':plan['separation_threshold'],
            'quantum_difference_max':max(row['quantum_readout_max_difference_FP32'] for row in rows),
            'quantum_difference_median':float(np.median([row['quantum_readout_max_difference_FP32'] for row in rows])),
            'classical_difference_max':max(row['classical_readout_max_difference_FP32'] for row in rows),
            'PennyLane_first_pair_error':error,'PennyLane_first_pair_feature_difference':float(np.abs(reference[0]-reference[1]).max()),
            'PennyLane_version':version,
            'interpretation':'The fixed26-dimensional projection discards a constructed direction that the circuit readout can access. This is a branch-specific synthetic representation example; the full classical backbone sees all inputs and can distinguish the pair. It does not prove test-set gains or a quantum-unique benefit.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parameters',type=Path,default=Path(__file__).with_name('projection_information_seed44.json'))
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('projection_information_result.json'))
    args=parser.parse_args()
    params=json.loads(args.parameters.read_text())
    plan={'status':'fixed_before_synthetic_outputs','seed':20261015,'pairs':16,'direction_l2':.7,
          'identity_tolerance':1e-10,'classical_FP32_tolerance':5e-5,'physical_reference_tolerance':5e-6,
          'separation_threshold':1e-4,'parameters_sha256':sha(args.parameters),
          'source_sha256':sha(__file__),'readout_source_sha256':sha(Path(__file__).with_name('quantum_readout.py')),
          'scope':'Exploratory deterministic synthetic branch-level representation audit; no dataset or classification claim.'}
    plan_path=args.output.with_name(args.output.stem+'_plan.json')
    if plan_path.exists(): assert json.loads(plan_path.read_text())==plan
    else: plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    result=execute(params,plan)
    result.update(plan_sha256=sha(plan_path),parameters_sha256=sha(args.parameters),confirmation_accessed=False)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','pairs','constraint_rank','nullspace_dimension',
                     'quantum_pairs_above_threshold','quantum_difference_max','quantum_difference_median',
                     'classical_difference_max','PennyLane_first_pair_error')}),flush=True)


if __name__=='__main__':main()
