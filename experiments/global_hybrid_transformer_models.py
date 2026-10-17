"""Grouped feature Transformer and globally projected coherent residual hybrids."""
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

RUN = Path(__file__).resolve().parent
STUDY = RUN
sys.path.insert(0, str(STUDY))
from multiscale_phase_models import make_model as window_model

ANCHOR = STUDY / 'top10_newdata/models/mlp_rank08_window359_mlp_q6_ln1_wd0'

def ry(angle):
    c,s=torch.cos(angle/2),torch.sin(angle/2)
    return torch.stack((torch.stack((c,-s)),torch.stack((s,c)))).to(torch.complex64)

def rz(angle):
    return torch.diag(torch.stack((torch.exp(-.5j*angle),torch.exp(.5j*angle))))

def apply_gate(state, gate, qubit, qubits):
    shaped=state.reshape(len(state),2**(qubits-qubit-1),2,2**qubit)
    return torch.einsum('ij,bhjl->bhil',gate,shaped).reshape(len(state),2**qubits)

class GlobalCoherentReadout(nn.Module):
    def __init__(self,qubits=6,depth=4):
        super().__init__()
        self.qubits,self.depth=qubits,depth
        self.theta=nn.Parameter(.1*torch.randn(depth,qubits,3))
        self.frequency=nn.Parameter(torch.full((depth,qubits),.25))
        self.shift=nn.Parameter(torch.zeros(depth,qubits))
        index=torch.arange(2**qubits)
        self.register_buffer('zsign',torch.stack([1-2*((index>>q)&1) for q in range(qubits)],-1).float())
        for q in range(qubits):
            target=(q+1)%qubits
            self.register_buffer(f'cnot_{q}',index ^ (((index>>q)&1)<<target))
            self.register_buffer(f'xflip_{q}',index ^ (1<<q))

    def forward(self,amplitude,angles):
        state=amplitude.to(torch.complex64)
        for layer in range(self.depth):
            for q in range(self.qubits):
                w=self.theta[layer,q]
                gate=rz(w[2])@ry(w[1])@rz(w[0])
                state=apply_gate(state,gate,q,self.qubits)
            for q in range(self.qubits):
                state=state[:,getattr(self,f'cnot_{q}')]
            phase=angles*self.frequency[layer]+self.shift[layer]
            # Batched diagonal RZ gates, with qubit0 as the least significant bit.
            state=state*torch.exp(-.5j*(phase@self.zsign.T))
        probability=state.real.square()+state.imag.square()
        z=probability@self.zsign
        xs,ys=[],[]
        for q in range(self.qubits):
            overlap=state.conj()*state[:,getattr(self,f'xflip_{q}')]
            xs.append(overlap.real.sum(-1))
            ys.append((overlap.imag*self.zsign[:,q]).sum(-1))
        return torch.cat((torch.stack(xs,-1),torch.stack(ys,-1),z),-1)

class GlobalResidualHybrid(nn.Module):
    def __init__(self,kind='quantum',qubits=6,depth=4,control_hidden=1,**unused):
        super().__init__()
        self.kind,self.qubits,self.depth=kind,qubits,depth
        config=json.loads((ANCHOR/'config.json').read_text())
        self.backbone=window_model(config['job']['spec'])
        self.backbone.load_state_dict(torch.load(ANCHOR/'best_macro.pt',map_location='cpu',weights_only=True))
        width=2**qubits
        self.projection=nn.Linear(359,width)
        self.normalization=nn.LayerNorm(width)
        if kind=='quantum':
            self.readout=GlobalCoherentReadout(qubits,depth)
        elif kind=='classical':
            self.readout=nn.Sequential(nn.Linear(width+qubits,control_hidden),nn.GELU(),
                                      nn.Linear(control_hidden,3*qubits))
        else:
            raise ValueError(kind)
        self.head=nn.Linear(3*qubits,10)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def encoded(self,x):
        latent=self.projection(torch.atan(x))
        amplitude=F.normalize(self.normalization(latent),p=2,dim=-1,eps=1e-8)
        bounded=torch.tanh(latent)
        group=math.ceil(bounded.shape[-1]/self.qubits)
        angles=F.pad(bounded,(0,group*self.qubits-bounded.shape[-1])).reshape(len(x),self.qubits,group).mean(-1)*math.pi
        return amplitude,angles

    def forward(self,x):
        amplitude,angles=self.encoded(x)
        output=self.readout(amplitude,angles) if self.kind=='quantum' else self.readout(torch.cat((amplitude,angles),-1))
        return self.backbone(x)+self.head(output)

    def counts(self):
        total=sum(p.numel() for p in self.parameters())
        gates=self.readout.theta.numel() if self.kind=='quantum' else 0
        encoding=self.readout.frequency.numel()+self.readout.shift.numel() if self.kind=='quantum' else 0
        return {'total':total,'quantum':gates,'quantum_encoding':encoding,'classical':total-gates-encoding,
                'pretrained_backbone':23170,'global_projection':self.projection.weight.numel()+self.projection.bias.numel(),
                'simultaneous_coherent_qubits':self.qubits if self.kind=='quantum' else 0,
                'circuit_depth':self.depth if self.kind=='quantum' else 0,'measurement':'physical X/Y/Z' if self.kind=='quantum' else 'classical MLP',
                'backbone_trainable':True,'all_parameters_active':True}

class BackboneContinuation(nn.Module):
    def __init__(self,**unused):
        super().__init__()
        config=json.loads((ANCHOR/'config.json').read_text())
        self.backbone=window_model(config['job']['spec'])
        self.backbone.load_state_dict(torch.load(ANCHOR/'best_macro.pt',map_location='cpu',weights_only=True))
    def forward(self,x):return self.backbone(x)
    def counts(self):return {'total':23170,'quantum':0,'quantum_encoding':0,'classical':23170,'pretrained_backbone':23170,'all_parameters_active':True}

class AttentionBlock(nn.Module):
    def __init__(self,dim,heads,ffn):
        super().__init__()
        self.dim,self.heads=dim,heads
        self.norm1,self.norm2=nn.LayerNorm(dim),nn.LayerNorm(dim)
        self.qkv=nn.Linear(dim,3*dim)
        self.out=nn.Linear(dim,dim)
        self.ffn=nn.Sequential(nn.Linear(dim,ffn),nn.GELU(),nn.Linear(ffn,dim))
    def forward(self,x):
        b,t,d=x.shape
        qkv=self.qkv(self.norm1(x)).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4)
        attended=F.scaled_dot_product_attention(qkv[0],qkv[1],qkv[2],dropout_p=0.)
        x=x+self.out(attended.transpose(1,2).reshape(b,t,d))
        return x+self.ffn(self.norm2(x))

class FeatureTransformer(nn.Module):
    def __init__(self,dim=48,heads=3,ffns=(192,192),head_hidden=48,**unused):
        super().__init__()
        assert dim%heads==0
        self.dim,self.heads=dim,heads
        self.token_weight=nn.Parameter(torch.empty(359,dim))
        self.token_bias=nn.Parameter(torch.zeros(120,dim))
        nn.init.normal_(self.token_weight,std=1/math.sqrt(3))
        self.blocks=nn.ModuleList([AttentionBlock(dim,heads,f) for f in ffns])
        self.final_norm=nn.LayerNorm(dim)
        self.head=nn.Sequential(nn.Linear(dim,head_hidden),nn.GELU(),nn.Linear(head_hidden,10))
        self.ffns,self.head_hidden=tuple(ffns),head_hidden
    def forward(self,x):
        feature_embeddings=torch.atan(x).unsqueeze(-1)*self.token_weight
        h=F.pad(feature_embeddings,(0,0,0,1)).reshape(len(x),120,3,self.dim).sum(2)+self.token_bias
        for block in self.blocks:h=block(h)
        return self.head(self.final_norm(h).mean(1))
    def counts(self):
        total=sum(p.numel() for p in self.parameters())
        expected=481*self.dim+10+sum(4*self.dim**2+9*self.dim+(2*self.dim+1)*f for f in self.ffns)+(self.dim+11)*self.head_hidden
        assert total==expected
        return {'total':total,'quantum':0,'quantum_encoding':0,'classical':total,
                'tokens':120,'features':359,'tokenization':'119 complete 3-feature groups plus final padded group',
                'embedding_dim':self.dim,'attention_heads':self.heads,'layers':len(self.blocks),
                'ffn_widths':list(self.ffns),'classifier_hidden':self.head_hidden,
                'dropout':0.,'all_parameters_active':True}

def transformer_spec(target):
    choices=[]
    for dim in (16,24,32,40,48,56,64,80,96,128):
        layers=2
        base=481*dim+10+layers*(4*dim*dim+9*dim)
        for head in range(8,4*dim+1):
            remainder=target-base-(dim+11)*head
            if remainder>0 and remainder%(2*dim+1)==0:
                total=remainder//(2*dim+1)
                widths=[total//layers+(i<total%layers) for i in range(layers)]
                if min(widths)>=dim and max(widths)<=8*dim:
                    score=abs(math.log(sum(widths)/layers/(4*dim)))+abs(math.log(head/dim))
                    choices.append((score,dim,widths,head))
    assert choices, target
    _,dim,widths,head=min(choices)
    heads=dim//16 if dim%16==0 and dim>=32 else dim//8
    return {'kind':'transformer','dim':dim,'heads':heads,'ffns':widths,'head_hidden':head}

def make_model(spec):
    if spec['kind']=='transformer':return FeatureTransformer(**spec)
    if spec['kind']=='backbone':return BackboneContinuation(**spec)
    return GlobalResidualHybrid(**dict(spec,kind='quantum' if spec['kind']=='globalquantum' else 'classical'))
