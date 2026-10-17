"""Standalone q9d12 readout, with qubit0 the least significant basis bit.

Extracted from the audited experiment's equations. There are no dataset paths,
pretrained backbones, network calls or credentials in this module.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def encode(x, qubits=9):
    """359 preprocessed inputs -> normalized amplitudes and nine group angles."""
    assert x.ndim == 2 and x.shape[-1] == 359 and qubits == 9
    raw = torch.atan(x)
    padded = F.pad(raw, (0, 2**qubits-359))
    norm = torch.linalg.vector_norm(padded, dim=-1, keepdim=True)
    zero_basis = torch.zeros_like(padded); zero_basis[:,0] = 1
    amplitude = torch.where(norm>1e-8, padded/norm.clamp_min(1e-8), zero_basis)
    group = math.ceil(359/qubits)
    angles = 2*F.pad(raw, (0, qubits*group-359)).reshape(len(x),qubits,group).mean(-1)
    return amplitude, angles


def ry(angle):
    c, s = torch.cos(angle/2), torch.sin(angle/2)
    return torch.stack((torch.stack((c,-s)),torch.stack((s,c)))).to(torch.complex64)


def rz(angle):
    return torch.diag(torch.stack((torch.exp(-.5j*angle),torch.exp(.5j*angle))))


def apply_gate(state, gate, qubit, qubits):
    shaped = state.reshape(len(state),2**(qubits-qubit-1),2,2**qubit)
    return torch.einsum('ij,bhjl->bhil',gate,shaped).reshape(len(state),2**qubits)


class QuantumReadout(nn.Module):
    def __init__(self, depth=12, cnot=True):
        super().__init__()
        self.qubits, self.depth, self.cnot = 9, depth, cnot
        self.theta = nn.Parameter(.1*torch.randn(depth,9,3))
        self.frequency = nn.Parameter(torch.ones(depth,9))
        self.shift = nn.Parameter(torch.zeros(depth,9))
        index = torch.arange(512)
        self.register_buffer('zsign',torch.stack([1-2*((index>>q)&1) for q in range(9)],-1).float())
        for q in range(9):
            self.register_buffer('cnot_'+str(q),index^(((index>>q)&1)<<((q+1)%9)))
            self.register_buffer('xflip_'+str(q),index^(1<<q))

    def forward(self, amplitude, angles):
        state = amplitude.to(torch.complex64)
        for layer in range(self.depth):
            for q in range(9):
                w = self.theta[layer,q]
                state = apply_gate(state,rz(w[2])@ry(w[1])@rz(w[0]),q,9)
            if self.cnot:
                for q in range(9):
                    state = state[:,getattr(self,'cnot_'+str(q))]
            phase = angles*self.frequency[layer]+self.shift[layer]
            state = state*torch.exp(-.5j*(phase@self.zsign.T))
        probability = state.real.square()+state.imag.square()
        z = probability@self.zsign
        xs, ys = [], []
        for q in range(9):
            overlap = state.conj()*state[:,getattr(self,'xflip_'+str(q))]
            xs.append(overlap.real.sum(-1))
            ys.append((overlap.imag*self.zsign[:,q]).sum(-1))
        return torch.cat((torch.stack(xs,-1),torch.stack(ys,-1),z),-1)

    def load_parameters(self, parameters):
        with torch.no_grad():
            for name in ('theta','frequency','shift'):
                value = torch.tensor(parameters[name],dtype=torch.float32)
                assert value.shape == getattr(self,name).shape
                getattr(self,name).copy_(value)

