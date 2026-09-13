import pickle
import json
import time
from pathlib import Path

import gymnasium as gym
import mani_skill.envs
import numpy as np
import torch
from scipy.fft import idct

from model import Config, VLA
from tokenizer import FAST

HORIZON = 10
FACTOR = 0.01
EPISODES = 10


def scalar(x): return bool(torch.as_tensor(x).any().item())


def predict(model, image, tok, device):
  image = torch.as_tensor(image, device=device)
  image_enc = model.encoder_img(image)
  tokens = torch.empty((image.shape[0], 0), dtype=torch.long, device=device)
  expected = HORIZON * 8
  decoded = []
  token_values = [tok.decode([token]) for token in range(tok.eos_id)]
  for _ in range(model.config.T - 1):
    logits = model(tokens, image_enc)[:, -1, :] / model.config.TEMP
    remaining = expected - len(decoded)
    valid = torch.zeros_like(logits, dtype=torch.bool)
    for token, values in enumerate(token_values):
      if len(values) <= remaining: valid[:, token] = True
    logits = logits.masked_fill(~valid, float("-inf"))
    next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
    tokens = torch.cat((tokens, next_token), dim=1)
    decoded.extend(token_values[next_token[0, 0].item()])
    if len(decoded) == expected: break
  if len(decoded) != expected:
    raise RuntimeError(f"Generated trajectory has {len(decoded)} values; expected {expected}")
  values = np.asarray(decoded)
  values = values + tok.min
  return idct(values.reshape(HORIZON, 8) * FACTOR, axis=0, norm="ortho").astype(np.float32)


def main():
  data = next(Path("demos").rglob("*.h5"))
  with open(data.with_suffix(".pkl"), "rb") as file: tok = pickle.load(file)
  with open(data.with_suffix(".json")) as file: seeds = [x["episode_seed"] for x in json.load(file)["episodes"][:EPISODES]]
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = VLA(tok, Config()).to(device)
  checkpoint = torch.load("vla.pt", map_location=device)
  model.load_state_dict(checkpoint["model"])
  model.eval()
  env = gym.make("PushCube-v1", obs_mode="rgb", control_mode="pd_joint_pos", render_mode="human", num_envs=1)
  successes = 0
  try:
    for episode, seed in enumerate(seeds):
      obs, _ = env.reset(seed=seed)
      env.render()
      done = False
      while not done:
        image = obs["sensor_data"]["base_camera"]["rgb"]
        with torch.inference_mode(): actions = predict(model, image, tok, device)
        for action in actions:
          start = time.perf_counter()
          obs, _, terminated, truncated, info = env.step(action[None])
          env.render()
          done = scalar(terminated) or scalar(truncated)
          time.sleep(max(0.0, 0.05 - (time.perf_counter() - start)))
          if done: break
      success = scalar(info.get("success", False))
      successes += success
      print(f"Episode {episode + 1}: {'success' if success else 'failure'} ({successes}/{episode + 1})")
  finally:
    env.close()


if __name__ == "__main__": main()
