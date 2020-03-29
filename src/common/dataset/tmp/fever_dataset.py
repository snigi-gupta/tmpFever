#!/usr/bin/env python

from torch.utils.data import Dataset
import json
from fever_example import FeverExample
import six

class FeverDataset(Dataset):

    def __init__(self, path):
      self.tv_datafields = [ 
                 ("claim", TEXT), ("label", LABEL),
                 ("evidence", TEXT)
                 ]
      self.path = path
      self.samples = self.get_lines(path)
      self.samples = self._convert_samples(self.samples, self.tv_datafields)
      super().__init__(self.samples, self.tv_datafields)
  
    def _convert_samples(self, samples, tv_datafields):
      conv = []
      for sample in samples:
        for eset in sample["evidence"]:
          for i in range(len(eset)):
            eset[i] = eset[i][3]
        # print(sample["evidence"])
        print("____________________________")
        cs = FeverExample.fromlist([sample["claim"],
                               self._label_to_int(sample["label"]),
                               sample["evidence"]],
                               tv_datafields)
        conv.append(cs)
      return conv

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
      return self.samples[idx]

    def _label_to_int(self, label):
      if label == "SUPPORTS":
        return 0
      if label == "REFUTES":
        return 1
      return 2

    def get_lines(self, path):
      with open(path, 'r') as f:
        return [json.loads(line) for line in f.readlines()[:10]]

#train_path = '/content/gdrive/My Drive/colab/635/group_project/train.jsonl'
train_path = '/home/kikuchio/git_repos/635-nlp-group-project/tmpFever/src/data/tmp/train_augmented.jsonl'
ds = FeverDataset(train_path)
for l in ds:
  print(l)
