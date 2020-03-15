import torch.nn as nn


class NeiRteModel(nn.Module):
    def forward(self, input):
        print("IN NEIRTE MODEL, received ", input)
        return "NOT_ENOUGH_INFO"