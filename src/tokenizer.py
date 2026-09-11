import h5py
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.fft import dct

class FAST:
  def __init__(self, merges) -> None:
    self.num_of_merges = merges
    self.vocab = {}
  def train_bpe(self, chunks: list, mi, ma):
    vocab = {}
    curId = ma - mi + 1
    for merge in tqdm(range(self.num_of_merges), desc="Training BPE"):
      localPairs = {}
      for chunk in chunks:
        for i in range(len(chunk)-1):
          pair = (chunk[i], chunk[i+1])
          if pair in localPairs: localPairs[pair] += 1
          else: localPairs[pair] = 1
      if not localPairs: break
      highest_key, highest_value = max(localPairs.items(), key=lambda item: item[1])
      mergeId = curId
      vocab[highest_key] = mergeId
      curId += 1
      for chunk_index in range(len(chunks)):
        new_chunk = []
        chunk = chunks[chunk_index]
        i = 0
        while i < len(chunk):
          if i + 1 < len(chunk) and (chunk[i], chunk[i + 1]) == highest_key:
            new_chunk.append(mergeId)
            i += 2
          else:
            new_chunk.append(chunk[i])
            i += 1
        chunks[chunk_index] = new_chunk
    self.vocab = vocab
    self.curId = curId
  def encode(self, chunk) -> list:
    for i in range(self.num_of_merges):
      new_chunk = []
      i = 0
      while i < len(chunk):
        if i + 1 < len(chunk):
          pair = (chunk[i], chunk[i + 1])
          if pair in self.vocab:
            new_chunk.append(self.vocab[pair])
            i += 2
          else:
            new_chunk.append(chunk[i])
            i += 1
        else:
          new_chunk.append(chunk[i])
          i += 1
      chunk = new_chunk
    return chunk
  def decode(self, chunk) -> list:
    new_chunk = []
    rev = {value: key for key, value in self.vocab.items()}
    for c in chunk:
      if c in rev: new_chunk.extend(self.decode(rev[c]))
      else: new_chunk.append(c)
    return new_chunk

FACTOR = 0.01
NUM_OF_MERGES = 500
HORIZON = 10

def build_tokenizer(file=None):
  file = Path(file) if file else next(Path("demos/PushCube-v1/motionplanning").glob("*.h5"))
  chunks = []
  with h5py.File(file) as source:
    for name in source:
      actions = source[name]["actions"][()]
      for start in range(0, len(actions) - HORIZON + 1, HORIZON):
        chunks.append(np.rint(dct(actions[start:start + HORIZON], axis=0, norm="ortho") / FACTOR).astype(np.int64).ravel().tolist())
  mi = min(min(chunk) for chunk in chunks)
  ma = max(max(chunk) for chunk in chunks)
  chunks = [[token - mi for token in chunk] for chunk in chunks]
  original_size = sum(map(len, chunks))
  fast = FAST(NUM_OF_MERGES)
  fast.train_bpe(chunks, mi, ma)
  compression = 100 * (1 - sum(map(len, chunks)) / original_size)
  print(f"Compression: {compression:.2f}%\nVocab size: {len(fast.vocab)}")
  with open(file.with_suffix(".pkl"), "wb") as output: pickle.dump(fast, output)
  return fast

build_tokenizer()
