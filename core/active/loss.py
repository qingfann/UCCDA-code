import torch
import torch.nn as nn


def bound_max_loss(energy, bound):
    energy_minus_bound = energy - bound
    energy_minus_bound = torch.unsqueeze(energy_minus_bound, dim=1)
    zeros = torch.zeros_like(energy_minus_bound)
    for_select = torch.cat((energy_minus_bound, zeros), dim=1)
    selected = torch.max(for_select, dim=1).values

    return selected.mean()


class EnergyLoss(nn.Module):

    def __init__(self, cfg):
        super(EnergyLoss, self).__init__()
        assert cfg.TRAINER.ENERGY_BETA > 0, "beta for energy calculate must be larger than 0"
        self.beta = cfg.TRAINER.ENERGY_BETA

        self.type = cfg.TRAINER.ENERGY_ALIGN_TYPE

        if self.type == 'l1':
            self.loss = nn.L1Loss()
        elif self.type == 'mse':
            self.loss = nn.MSELoss()
        elif self.type == 'max':
            self.loss = bound_max_loss

    def forward(self, inputs, average):
        mul_neg_beta = -1.0 * self.beta * inputs
        log_sum_exp = torch.logsumexp(mul_neg_beta, dim=1)
        energies = -1.0 * log_sum_exp / self.beta
        average = torch.ones_like(energies) * average
        Enloss = self.loss(energies, average)
        return Enloss

