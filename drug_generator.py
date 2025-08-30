import gymnasium as gym
import math
import matplotlib.pyplot as plt
import mygene
import numpy as np
import os
import pandas as pd
import random
import selfies as sf
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from prot_autoencoder import ProtEncoder
from prot_ligand_model import ProtLigandModel
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Descriptors
from rdkit.Contrib.SA_Score import sascorer
from selfies_autoencoder import SelfiesDecoder
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from transformers import AutoTokenizer, AutoModel
from utils import embedding_to_smiles, prot_to_embedding

def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def validate_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        Chem.SanitizeMol(mol)  # This will raise an exception if the molecule is invalid
        return True
    except:
        return False

def get_qed_score(smiles):
    mol = Chem.MolFromSmiles(smiles)
    qed_score = QED.qed(mol)
    return qed_score

def get_sa_score(smiles):
    mol = Chem.MolFromSmiles(smiles)
    sa_score = sascorer.calculateScore(mol)
    return sa_score

def get_toxicity_score(smiles):
    pass

class DrugGenEnv(gym.Env):
    def __init__(self):
        super().__init__()
        with open("Data/selfies_alphabet.txt", "r") as f:
            self.selfies_alphabet = f.read().splitlines()
        
        with open("Data/prot_alphabet.txt", "r") as f:
            self.prot_alphabet = f.read().splitlines()

        selfies_ae_checkpoint = torch.load("Models/selfies_autoencoder.pth")
        self.selfies_ae_hidden_size = selfies_ae_checkpoint["embedding_dim"]
        selfies_chunk_len = selfies_ae_checkpoint["chunk_length"]
        self.selfies_decoder = SelfiesDecoder(len(self.selfies_alphabet),
                                selfies_chunk_len=selfies_chunk_len,
                                hidden_size=self.selfies_ae_hidden_size)
        self.selfies_decoder.load_state_dict(selfies_ae_checkpoint["decoder_model"])
        self.selfies_decoder.eval()

        prot_ae_checkpoint = torch.load("Models/prot_autoencoder.pth")
        prot_ae_hidden_size = prot_ae_checkpoint["embedding_dim"]
        prot_chunk_len = prot_ae_checkpoint["chunk_length"]
        self.prot_encoder = ProtEncoder(len(self.prot_alphabet), prot_chunk_len=prot_chunk_len, hidden_size=prot_ae_hidden_size)
        self.prot_encoder.load_state_dict(prot_ae_checkpoint["encoder_model"])

        prot_ligand_checkpoint = torch.load("Models/prot_ligand_model.pth")
        prot_ligand_hidden_size = prot_ligand_checkpoint["hidden_size"]
        max_selfies_len = prot_ligand_checkpoint["max_selfies_len"]
        max_prot_len = prot_ligand_checkpoint["max_prot_len"]
        selfies_embed_dim = (max_selfies_len // selfies_chunk_len) * self.selfies_ae_hidden_size
        prot_embed_dim = (max_prot_len // prot_chunk_len) * prot_ae_hidden_size
        self.prot_ligand_model = ProtLigandModel(prot_embed_dim, selfies_embed_dim, prot_ligand_hidden_size)
        self.prot_ligand_model.load_state_dict(prot_ligand_checkpoint["model_state_dict"])

        self.viral_prot_df = pd.read_csv("Data/virus_prots.tsv", sep="\t", index_col="Entry", nrows=200)
        self.viral_prot_df = self.viral_prot_df[self.viral_prot_df["Sequence"].str.len() <= max_prot_len]
        self.viral_prot_df["Embedding"] = self.viral_prot_df.apply(lambda x: prot_to_embedding(x["Sequence"], self.prot_alphabet, self.prot_encoder, prot_chunk_len, max_prot_len).flatten(), axis=1)
        self.viral_prot_df["Reward"] = 0

        self.max_selfies_len = max_selfies_len
        self.reward_list = []

        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], label="Reward")
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Reward")
        self.ax.set_title("Reward Over Time")
        self.ax.legend()
        self.fig.show()
        self.fig.canvas.draw()

        self.observation_space = spaces.Box(low=0, high=1, shape=(prot_embed_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(selfies_embed_dim,), dtype=np.float32)

    def step(self, action: np.ndarray):
        prot_embedding = self.current_prot_data.iloc[0]["Embedding"]
        selfies_embedding_flat = action
        selfies_embedding_unflat = action.reshape(-1, self.selfies_ae_hidden_size)
        smiles = embedding_to_smiles(selfies_embedding_unflat, self.selfies_alphabet, self.selfies_decoder)
        if not validate_molecule(smiles):
            return self.current_prot_data.iloc[0]["Embedding"], 0, True, False, {}
        
        qed_score = get_qed_score(smiles)
        sa_score = get_sa_score(smiles)
        # risk_score = rapid_toxicity_screen(smiles)

        with torch.no_grad():
            prot_tensor = torch.tensor(prot_embedding, dtype=torch.float32).unsqueeze(0)
            selfies_tensor = torch.tensor(selfies_embedding_flat, dtype=torch.float32).unsqueeze(0)
            binding_aff = self.prot_ligand_model(prot_tensor, selfies_tensor)[0].item()

        reward = max(0, binding_aff) * sigmoid(10 * (qed_score - 0.6)) * sigmoid(10 * (0.4 - sa_score / 10))
        self.reward_list.append(reward)

        prot_entry = self.current_prot_data.index[0]
        if reward > self.viral_prot_df.loc[prot_entry, "Reward"]:
            organism = self.current_prot_data.iloc[0]["Organism"]
            print(f"Protein ID: {prot_entry}")
            print(f"Organism: {organism}")
            print(f"SMILES: {smiles}")
            print(f"Binding Affinity: {binding_aff}")
            print(f"Drug QED Score: {qed_score}")
            print(f"Drug SA Score: {sa_score}")
            print(f"Reward: {reward}")
            
            self.viral_prot_df.loc[prot_entry, "Reward"] = reward

        return  self.current_prot_data.iloc[0]["Embedding"], reward, True, False, {}

    def reset(self, seed=None, options=None):
        self.current_prot_data = self.viral_prot_df.sample()

        if len(self.reward_list) % 100 == 0 and len(self.reward_list) > 0:
            reward_list_smooth = moving_average(self.reward_list)
            self.line.set_data(range(len(reward_list_smooth)), reward_list_smooth)

            self.ax.relim()
            self.ax.autoscale_view()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            plt.pause(0.01)

        return self.current_prot_data.iloc[0]["Embedding"], {}

def main():    
    policy_savefile = "Models/drug_generator"

    env = DrugGenEnv()
    policy_kwargs = dict(
        net_arch=[1200, 1200],
        activation_fn=torch.nn.GELU
    )

    model = PPO("MlpPolicy", env, n_steps=512, batch_size=128, n_epochs=5, learning_rate=1e-5, ent_coef=1e-3, policy_kwargs=policy_kwargs)
    if os.path.exists(f"{policy_savefile}.zip"):
        model.set_parameters(policy_savefile)

    for epoch in range(100):
        model.learn(total_timesteps=5000, progress_bar=True)
        model.save(policy_savefile)

    plt.ioff()
    plt.close()

if __name__=="__main__":
    main()