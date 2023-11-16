from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Consensus_dictionaryLoss(nn.Module):

    def __init__(self, nav_t: float, beta: float, num_classes: int, device: torch.device, s_par: Optional[float] = 0.5,
                 reduction: Optional[str] = 'mean'):
        super(Consensus_dictionaryLoss, self).__init__()
        self.nav_t = nav_t
        self.s_par = s_par
        self.beta = beta
        self.prop = (torch.ones((num_classes, 1)) * (1 / num_classes)).to(device)
        self.eps = 1e-6

    def M_distance(self, x, y):
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        Distance=1 - torch.matmul(x, y.T)
        return Distance

    def UCCDA_logits(self, sim_mat, prop):
        log_prior = torch.log(prop + self.eps)
        return sim_mat / self.nav_t + log_prior

    def update_prop(self, prop):
         Update=(1 - self.beta) * self.prop + self.beta * prop
         return Update

    def forward(self, mu_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        # Update
        sim_mat = torch.matmul(mu_s, f_t.T)
        old_logits = self.UCCDA_logits(sim_mat.detach(), self.prop)
        s_dist_old = -F.softmax(-old_logits, dim=0)
        prop = s_dist_old.mean(1, keepdim=True)
        self.prop = self.update_prop(prop)

        # Calculate  transport loss
        new_logits = self.UCCDA_logits(sim_mat, self.prop)
        s_dist = -F.softmax(-new_logits, dim=0)
        t_dist = -F.softmax(-sim_mat / self.nav_t, dim=1)

        cost_mat = self.M_distance(mu_s, f_t)
        source_loss = (self.s_par * cost_mat * s_dist).sum(0).mean()
        target_loss = (((1 - self.s_par) * cost_mat * t_dist).sum(1) * self.prop.squeeze(1)).sum()
        Align_loss = source_loss + target_loss
        return Align_loss