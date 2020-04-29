import torch
import torch.nn as nn


class LabelAggregator(nn.Module):

    label_int_val = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
    label_str_val = {i: s for s, i in label_int_val.items()}

    def __init__(self, num_labels=3, num_sentences=5):
        super().__init__()
        input_dim = num_labels*num_sentences*2
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, input_dim)
        self.fc4 = nn.Linear(input_dim, num_labels)
        self.dout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(True)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialise the weights of the Aggregator model.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            nn.init.constant_(module.bias.data, 0.0)

    def forward(self, sen_logits):
        x = self.relu(self.fc1(sen_logits))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        logits = self.fc4(x)
        probabilities = nn.functional.softmax(logits, dim=-1)
        return logits, probabilities

    @staticmethod
    def from_state(state_path, device) -> 'LabelAggregator':
        inst = LabelAggregator(device)
        inst.load_state_dict(torch.load(state_path, map_location=device))
        return inst

    def save_state(self, save_path) -> 'LabelAggregator':
        torch.save(self.state_dict(), save_path)
        return self

    def _label_str_to_int(self, label: str) -> int:
        return self.label_int_val[label]

    def _label_int_to_str(self, label: int) -> str:
        return self.label_str_val[label]
    

