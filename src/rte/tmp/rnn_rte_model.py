import torch
import torch.nn as nn


class RnnRteModel(nn.Module):

    label_int_val = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
    label_str_val = { i: s for s, i in label_int_val.items()}

    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, claim: str, sentences: list) -> str:
        pred = 0
        print("IN RNN MODEL, received ", claim, sentences)
        return self._label_int_to_str(pred) 

    @staticmethod
    def from_state(state_path, device) -> 'RnnRteModel':
        inst = RnnRteModel(device)
        inst.load_state_dict(torch.load(state_path, map_location=device))
        return inst

    def save_state(self, save_path) -> 'RnnRteModel':
        torch.save(self.state_dict(), save_path)
        return self

    def _label_str_to_int(self, label: str) -> int:
        return self.label_int_val[label]

    def _label_int_to_str(self, label: int) -> str:
        return self.label_str_val[label]
    

