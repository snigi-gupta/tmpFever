# from torchtext import datasets
from torch.utils.data import Dataset
#from torchtext import data
import json

# train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
class FeverDataset(Dataset):

  def __init__(self, path):
    self.path = path
    self.samples = self.get_lines(path)
  
  def __len__(self):
      return len(self.samples)

  def __getitem__(self, idx):
    sample = self.samples[idx]
    return (sample, self._label_to_int(sample["label"]))

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
