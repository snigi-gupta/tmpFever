import torch.nn as nn


class NeiRteModel(nn.Module):

    def __init__(self, device):
        super().__init__()
    
    def forward(self, claim: str, sentences: list) -> str:
        print("IN NEIRTE MODEL, received ", claim, sentences)
        return "NOT ENOUGH INFO"

    @staticmethod
    def from_state(state_path, device) -> 'NeiRteModel':
        return NeiRteModel(device)

    def save_state(self, state_path) -> 'NeiRteModel':
        return self

