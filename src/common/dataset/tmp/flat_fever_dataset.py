from torchtext.data import Dataset
from torchtext.data import Example
import json
import six


class FlatFeverDataset(Dataset):

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
        all_ev = ""
        for eset in sample["evidence"]:
          for i in range(len(eset)):
            eset[i] = eset[i][3]
            all_ev += eset[i] + " "
        
        cs = Example.fromlist([sample["claim"],
                               self._label_to_int(sample["label"]),
                               all_ev],
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
