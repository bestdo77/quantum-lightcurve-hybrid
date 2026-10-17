"""Export only audited seed44 readouts for a public synthetic representation check."""
import json
import sys
from pathlib import Path
import torch

ROOT=Path('/mnt/nvme/home/yxh/gtp_qml_search_20261009')
sys.path.insert(0,str(ROOT))
import paper_evidence_neural as neural
from train_priority_14235 import sha,write

PUBLIC=Path('/mnt/hdd/yxh/qml_paper_evidence_20261014/projection_information_public')
PUBLIC.mkdir(parents=True,exist_ok=True)
parameters={'scope':'Only quantum and matched-classical readouts; no data/backbone/credentials.',
            'selection':'Both seed44 best_macro validation checkpoints; exploratory mechanism example.',
            'quantum':{},'classical':{},'checkpoint_sha256':{}}
for kind in ('quantum','matched_mlp'):
    folder=neural.RUN/'models'/(kind+'_s44')
    audit=json.loads((folder/'result_audit.json').read_text())
    protocol=json.loads((folder/'protocol_audit.json').read_text())
    checkpoint=folder/'best_macro.pt'
    assert audit['status']==protocol['status']=='passed'
    assert audit['best_macro']['checkpoint_sha256']==sha(checkpoint)
    assert protocol['result_audit_sha256']==sha(folder/'result_audit.json')
    model=neural.make_model({'kind':kind,'seed':44}).eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    parameters['checkpoint_sha256'][kind]=sha(checkpoint)
    if kind=='quantum':
        parameters['quantum']={name:getattr(model.readout,name).detach().numpy().tolist()
                               for name in ('theta','frequency','shift')}
    else:
        parameters['classical']={'projection':model.readout.projection.numpy().tolist(),
                                'first_weight':model.readout.first.weight.detach().numpy().tolist(),
                                'first_bias':model.readout.first.bias.detach().numpy().tolist(),
                                'head_weight':torch.cat((model.readout.fast_head,model.readout.slow_head),1).detach().numpy().tolist()}
write(PUBLIC/'projection_information_seed44.json',parameters)
print(json.dumps({'status':'exported_audited_readouts','parameters_sha256':sha(PUBLIC/'projection_information_seed44.json'),
                  'checkpoint_sha256':parameters['checkpoint_sha256'],'dataset_accessed':False}),flush=True)
