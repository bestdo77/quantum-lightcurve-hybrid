"""Canonical AdamW updates, without dividing by LR in graph-captured warmup."""
import torch

class SafeCapturableAdamW(torch.optim.AdamW):
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            assert not group.get('amsgrad',False) and not group.get('maximize',False)
            beta1,beta2=group['betas']
            lr,decay,eps=group['lr'],group['weight_decay'],group['eps']
            for p in group['params']:
                if p.grad is None: continue
                assert not p.grad.is_sparse and not p.is_complex()
                state=self.state[p]
                if not state:
                    state.update(step=torch.zeros((),device=p.device),
                        exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
                state['step'].add_(1)
                state['exp_avg'].lerp_(p.grad,1-beta1)
                state['exp_avg_sq'].mul_(beta2).addcmul_(p.grad,p.grad,value=1-beta2)
                bias1=1-beta1**state['step']
                bias2=1-beta2**state['step']
                denominator=(state['exp_avg_sq'].sqrt()/bias2.sqrt()).add_(eps)
                p.mul_(1-lr*decay)
                p.add_(-(lr/bias1)*(state['exp_avg']/denominator))
        return loss
