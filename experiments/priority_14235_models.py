"""Quantum models for the user-prioritized 1 -> 4 -> 2 -> 3 -> 5 study."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from representation_fusion_models import WindowFeatures
from global_hybrid_transformer_models import GlobalCoherentReadout, apply_gate, ry, rz

def _encode(x, qubits, permutation, zero_basis):
    raw = torch.atan(x)[:, permutation]
    padded = F.pad(raw, (0, 2**qubits - raw.shape[-1]))
    norm = torch.linalg.vector_norm(padded, dim=-1, keepdim=True)
    amplitude = torch.where(norm > 1e-8, padded / norm.clamp_min(1e-8), zero_basis.unsqueeze(0))
    group = math.ceil(raw.shape[-1] / qubits)
    angles = 2 * F.pad(raw, (0, qubits * group - raw.shape[-1])).reshape(len(x), qubits, group).mean(-1)
    return amplitude, angles

class MatchedClassicalReadout(nn.Module):
    """Nonlinear classical readout with exactly the q9/d12 circuit's 540 parameters."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(27, 20))
        nn.init.normal_(self.weight, std=0.1)
        g = torch.Generator().manual_seed(20261013)
        self.register_buffer('indices', torch.randint(0, 359, (27, 20), generator=g))
    def forward(self, x):
        selected = torch.atan(x)[:, self.indices]
        return torch.tanh((selected * self.weight.unsqueeze(0)).sum(-1) / math.sqrt(20.0))

class ProgressiveReadout(GlobalCoherentReadout):
    """Read a q9 circuit at intermediate depths with identity-paired additions."""
    def __init__(self, qubits=9, depth=18):
        assert qubits == 9 and depth in (18, 24)
        super().__init__(qubits, depth)
        with torch.no_grad():
            self.frequency.fill_(1.0)
            self.theta[12:].zero_()
            self.frequency[12:].zero_()
            self.shift[12:].zero_()
    def _measure(self, state):
        probability = state.real.square() + state.imag.square()
        z = probability @ self.zsign
        xs, ys = [], []
        for q in range(self.qubits):
            overlap = state.conj() * state[:, getattr(self, f'xflip_{q}')]
            xs.append(overlap.real.sum(-1))
            ys.append((overlap.imag * self.zsign[:, q]).sum(-1))
        return torch.cat((torch.stack(xs, -1), torch.stack(ys, -1), z), -1)
    def forward(self, amplitude, angles):
        state = amplitude.to(torch.complex64)
        outputs = {}
        for layer in range(self.depth):
            for q in range(self.qubits):
                w = self.theta[layer, q]
                state = apply_gate(state, rz(w[2]) @ ry(w[1]) @ rz(w[0]), q, self.qubits)
            entangler_order = range(self.qubits)
            if layer >= 12 and (layer - 12) % 2 == 1:
                entangler_order = reversed(range(self.qubits))
            for q in entangler_order:
                state = state[:, getattr(self, f'cnot_{q}')]
            phase = angles * self.frequency[layer] + self.shift[layer]
            state = state * torch.exp(-0.5j * (phase @ self.zsign.T))
            d = layer + 1
            if d in (6, 12, 18, 24):
                outputs[d] = self._measure(state)
        return outputs

class PriorityQuantumModel(nn.Module):
    def __init__(self, variant, qubits=9, depth=12, registers=1, fusion_hidden=128, module_dropout=0.2):
        super().__init__()
        assert qubits == 9
        self.variant, self.qubits, self.depth, self.registers = variant, qubits, depth, registers
        self.module_dropout = module_dropout
        self.backbone = WindowFeatures()
        self.classical_norm = nn.LayerNorm(self.backbone.dim)
        basis = torch.zeros(2**qubits); basis[0] = 1.0
        self.register_buffer('zero_basis', basis)
        if variant in ('adamw_branchfirst', 'quantum_residual'):
            self.readout = GlobalCoherentReadout(qubits, depth)
            with torch.no_grad(): self.readout.frequency.fill_(1.0)
            self.register_buffer('feature_permutations', torch.arange(359).reshape(1, -1))
            self.representation_norm = nn.LayerNorm(27)
            self.fusion_head = self._head(self.backbone.dim + 27, fusion_hidden, True)
        elif variant == 'classical_residual_control':
            self.readout = MatchedClassicalReadout()
            self.representation_norm = nn.LayerNorm(27)
            self.fusion_head = self._head(self.backbone.dim + 27, fusion_hidden, True)
        elif variant == 'progressive':
            self.readout = ProgressiveReadout(qubits, depth)
            self.register_buffer('feature_permutations', torch.arange(359).reshape(1, -1))
            self.representation_norm = nn.LayerNorm(27)
            checkpoints = [d for d in (6, 18, 24) if d <= depth]
            self.checkpoint_norms = nn.ModuleDict({str(d): nn.LayerNorm(27) for d in checkpoints})
            self.residual_gates = nn.ParameterDict({str(d): nn.Parameter(torch.zeros(27)) for d in checkpoints})
            self.fusion_head = self._head(self.backbone.dim + 27, fusion_hidden, True)
        elif variant == 'heterogeneous':
            assert registers == 4 and depth == 6
            self.readouts, permutations = nn.ModuleList(), []
            for r in range(registers):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(42 + 97*r); readout = GlobalCoherentReadout(qubits, depth)
                with torch.no_grad():
                    readout.frequency.fill_(1.0 + 0.05*r); readout.shift.add_(0.01*r)
                self.readouts.append(readout)
                g = torch.Generator().manual_seed(20261013 + 1009*r)
                permutations.append(torch.randperm(359, generator=g) if r else torch.arange(359))
            self.register_buffer('feature_permutations', torch.stack(permutations))
            self.module_norms = nn.ModuleList([nn.LayerNorm(27) for _ in range(registers)])
            self.representation_norm = nn.LayerNorm(27*registers)
            self.fusion_head = self._head(self.backbone.dim + 27*registers, 256, True)
        elif variant == 'sparse_experts':
            assert registers == 4 and depth == 4
            self.readouts, permutations = nn.ModuleList(), []
            for r in range(registers):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(14235 + 131*r); readout = GlobalCoherentReadout(qubits, depth)
                with torch.no_grad(): readout.frequency.fill_(1.0 + 0.04*r)
                self.readouts.append(readout)
                g = torch.Generator().manual_seed(14235 + 997*r)
                permutations.append(torch.randperm(359, generator=g))
            self.register_buffer('feature_permutations', torch.stack(permutations))
            self.expert_norms = nn.ModuleList([nn.LayerNorm(27) for _ in range(registers)])
            self.expert_heads = nn.ModuleList([nn.Linear(27, 10) for _ in range(registers)])
            for head in self.expert_heads:
                nn.init.normal_(head.weight, std=1e-3); nn.init.zeros_(head.bias)
            self.gate = nn.Sequential(nn.Linear(self.backbone.dim + 2, 64), nn.GELU(), nn.Linear(64, registers))
            masks = torch.zeros(registers, 10)
            for r, classes in enumerate(((0,4,8),(1,3,6),(2,5,9),(3,7,9))): masks[r, list(classes)] = 1.0
            self.register_buffer('expert_masks', masks)
        else:
            raise ValueError(variant)
    @staticmethod
    def _head(inputs, hidden, zero_last):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42); head = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, 10))
        if zero_last:
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)
        return head
    def _quantum_feature(self, x, readout, permutation):
        amplitude, angles = _encode(x, self.qubits, permutation, self.zero_basis)
        return readout(amplitude, angles)
    def forward(self, x, return_aux=False):
        hidden = self.backbone.features(x)
        base = self.backbone.network.head(hidden)
        classical = self.classical_norm(hidden)
        aux = {'base': base}
        if self.variant in ('adamw_branchfirst', 'quantum_residual'):
            feature = self.representation_norm(self._quantum_feature(x, self.readout, self.feature_permutations[0]))
            residual = self.fusion_head(torch.cat((classical, feature), -1))
        elif self.variant == 'classical_residual_control':
            feature = self.representation_norm(self.readout(x))
            residual = self.fusion_head(torch.cat((classical, feature), -1))
        elif self.variant == 'progressive':
            amplitude, angles = _encode(x, self.qubits, self.feature_permutations[0], self.zero_basis)
            measured = self.readout(amplitude, angles)
            feature = self.representation_norm(measured[12])
            for key, gate in self.residual_gates.items():
                feature = feature + torch.tanh(gate)*self.checkpoint_norms[key](measured[int(key)])
            residual = self.fusion_head(torch.cat((classical, feature), -1))
            aux['checkpoint_features'] = measured
        elif self.variant == 'heterogeneous':
            modules = [self.module_norms[r](self._quantum_feature(x, readout, self.feature_permutations[r])) for r,readout in enumerate(self.readouts)]
            module_features = torch.stack(modules, 1)
            used = module_features
            if self.training and self.module_dropout:
                used = F.dropout1d(used, p=self.module_dropout, training=True)
            residual = self.fusion_head(torch.cat((classical, self.representation_norm(used.reshape(len(x),-1))), -1))
            aux['module_features'] = module_features
        else:
            modules, expert_logits = [], []
            for r,readout in enumerate(self.readouts):
                feature = self.expert_norms[r](self._quantum_feature(x, readout, self.feature_permutations[r]))
                modules.append(feature); expert_logits.append(self.expert_heads[r](feature)*self.expert_masks[r])
            base_prob = base.detach().softmax(-1)
            confidence = base_prob.max(-1).values.unsqueeze(-1)
            entropy = -(base_prob*base_prob.clamp_min(1e-8).log()).sum(-1,keepdim=True)/math.log(10.0)
            gates = self.gate(torch.cat((classical,confidence,entropy),-1)).softmax(-1)
            residual = (gates.unsqueeze(-1)*torch.stack(expert_logits,1)).sum(1)
            aux['module_features'] = torch.stack(modules,1); aux['gates'] = gates
        logits = base + residual
        aux['residual'] = residual
        return (logits, aux) if return_aux else logits
    def counts(self):
        total = sum(p.numel() for p in self.parameters())
        if hasattr(self,'readouts'):
            quantum = sum(r.theta.numel() for r in self.readouts)
            encoding = sum(r.frequency.numel()+r.shift.numel() for r in self.readouts)
        elif hasattr(self.readout,'theta'):
            quantum = self.readout.theta.numel(); encoding = self.readout.frequency.numel()+self.readout.shift.numel()
        else:
            quantum = encoding = 0
        return {'total':total,'quantum':quantum,'quantum_encoding':encoding,'classical':total-quantum-encoding,
            'pretrained_backbone':23170,'simultaneous_coherent_qubits':self.qubits if quantum else 0,
            'independent_registers':self.registers if quantum else 0,'circuit_depth':self.depth if quantum else 0,
            'variant':self.variant,'all_parameters_trainable_or_explicitly_frozen_by_protocol':True}

def make_model(spec):
    return PriorityQuantumModel(**spec)
