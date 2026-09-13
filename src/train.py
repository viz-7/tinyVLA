import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.fft import dct
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from model import Config, VLA
from tokenizer import FAST

HORIZON = 10
FACTOR = 0.01
EPOCHS = 500
LR = 3e-4


class Demos(Dataset):
  def __init__(self, path, tok):
    self.file = h5py.File(path)
    self.tok = tok
    self.items = [(name, start) for name in self.file for start in range(0, len(self.file[name]["actions"]) - HORIZON + 1, HORIZON)]

  def __len__(self): return len(self.items)

  def __getitem__(self, index):
    name, start = self.items[index]
    group = self.file[name]
    actions = group["actions"][start:start + HORIZON]
    values = np.rint(dct(actions, axis=0, norm="ortho") / FACTOR).astype(np.int64).ravel() - self.tok.min
    tokens = torch.tensor(self.tok.encode(values.tolist()) + [self.tok.eos_id], dtype=torch.long)
    image = torch.from_numpy(group["obs/sensor_data/base_camera/rgb"][start])
    return image, tokens


def collate(batch, pad_id):
  images, tokens = zip(*batch)
  return torch.stack(images), pad_sequence(tokens, batch_first=True, padding_value=pad_id)


def main():
  data = next(Path("demos").rglob("*.h5"))
  with open(data.with_suffix(".pkl"), "rb") as file: tok = pickle.load(file)
  loader = DataLoader(Demos(data, tok), batch_size=64, shuffle=True,
                      collate_fn=lambda batch: collate(batch, tok.pad_id))
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = VLA(tok, Config()).to(device)
  compiled = torch.compile(model, dynamic=True)
  optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
  epoch = 0
  try:
    for epoch in range(1, EPOCHS + 1):
      total = 0.0
      bar = tqdm(loader, desc=f"Epoch {epoch}")
      for image, tokens in bar:
        image, tokens = image.to(device), tokens.to(device)
        logits = compiled(tokens[:, :-1], compiled.encoder_img(image))
        loss = F.cross_entropy(logits.flatten(0, 1), tokens.flatten(), ignore_index=tok.pad_id)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")
      torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "loss": total / len(loader)}, "vla.pt")
  except KeyboardInterrupt:
    print("\nInterrupted")
  finally:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, "vla.pt")
    print("Saved vla.pt")


if __name__ == "__main__": main()
