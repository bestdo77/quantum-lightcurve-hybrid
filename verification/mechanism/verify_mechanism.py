"""Reproduce the local-information example and an independent physical check."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from quantum_readout import QuantumReadout, encode


def physical_check(readout, amplitude, angles):
    import pennylane as qml
    from pennylane import numpy as pnp
    coeff = np.linspace(-.7,.9,27);coeff /= np.linalg.norm(coeff)
    amplitude_tensor, angle_tensor = amplitude[:1], angles[:1]
    amplitude = amplitude[0].numpy().astype(float)
    amplitude /= np.linalg.norm(amplitude)
    angle = angles[0].numpy().astype(float)
    device = qml.device('default.qubit',wires=9)
    operators = [op(q) for op in (qml.PauliX,qml.PauliY,qml.PauliZ) for q in range(9)]
    @qml.qnode(device,interface='autograd',diff_method='adjoint')
    def circuit(theta,frequency,shift):
        qml.AmplitudeEmbedding(amplitude,wires=list(reversed(range(9))),normalize=True)
        for layer in range(readout.depth):
            for q in range(9):
                qml.RZ(theta[layer,q,0],wires=q)
                qml.RY(theta[layer,q,1],wires=q)
                qml.RZ(theta[layer,q,2],wires=q)
            if readout.cnot:
                for q in range(9):qml.CNOT(wires=[q,(q+1)%9])
            for q in range(9):qml.RZ(angle[q]*frequency[layer,q]+shift[layer,q],wires=q)
        return qml.expval(qml.dot(coeff,operators))
    params = (readout.theta,readout.frequency,readout.shift)
    reference_params = [pnp.array(p.detach().numpy().astype(float),requires_grad=True) for p in params]
    reference = qml.grad(circuit,argnum=(0,1,2))(*reference_params)
    # Use the same original float32 encoding that the readout consumes.
    objective = (readout(amplitude_tensor,angle_tensor)[0]*torch.tensor(coeff,dtype=torch.float32)).sum()
    actual = torch.autograd.grad(objective,params)
    errors = {name:float(np.max(np.abs(a.numpy()-np.asarray(b))))
              for name,a,b in zip(('theta','frequency','shift'),actual,reference)}
    forward_error = abs(float(objective)-float(circuit(*reference_params)))
    assert max(errors.values()) < 2e-5 and forward_error < 5e-6
    return {'derivative_max_errors':errors,'forward_error':forward_error,
            'readout_parameters_checked':sum(p.numel() for p in params),'pennylane_version':qml.__version__}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters',type=Path,default=Path(__file__).with_name('readout_initial_seed42.json'))
    parser.add_argument('--physical',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    parameters = json.loads(args.parameters.read_text())
    x = torch.zeros(2,359);x[:,0]=1;x[0,255]=1;x[1,255]=-1
    amplitude,angles = encode(x)
    results = []
    for cnot in (True,False):
        model = QuantumReadout(cnot=cnot)
        model.load_parameters(parameters)
        with torch.no_grad():output = model(amplitude,angles)
        difference = float((output[0]-output[1]).abs().max())
        assert difference > .01 if cnot else difference < 2e-6
        result = {'cnot':cnot,'maximum_feature_difference':difference}
        if args.physical:result['physical_check']=physical_check(model,amplitude,angles)
        results.append(result)
    print(json.dumps({'status':'passed','scope':'Synthetic representation example, not a classification benchmark.',
                      'torch_version':torch.__version__,'results':results},indent=2))


if __name__=='__main__':main()
