import numpy as np
import os
import pandas as pd
import random
import selfies as sf
import torch
import torch.nn as nn
import torch.optim as optim
from prot_autoencoder import ProtEncoder
from scipy.stats import pearsonr, spearmanr
from selfies_autoencoder import SelfiesEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset, Dataset, DataLoader
from utils import set_seeds, smiles_to_embedding, prot_to_embedding

class ProtLigandDataset(Dataset):
    def __init__(self, max_prot_len=1000, max_selfies_len=100, max_entries_per_prot=100, max_lines=200000):
        selfies_alphabet_path = "Data/selfies_alphabet.txt"
        with open(selfies_alphabet_path, "r") as f:
            selfies_alphabet = f.read().splitlines()
        
        prot_alphabet_path = "Data/prot_alphabet.txt"
        with open(prot_alphabet_path, "r") as f:
            prot_alphabet = f.read().splitlines()

        selfies_ae_path = "Models/selfies_autoencoder.pth"
        selfies_ae_checkpoint = torch.load(selfies_ae_path)
        selfies_embed_dim = selfies_ae_checkpoint["embedding_dim"]
        selfies_chunk_len = selfies_ae_checkpoint["chunk_length"]
        selfies_encoder = SelfiesEncoder(len(selfies_alphabet), selfies_chunk_len=selfies_chunk_len, hidden_size=selfies_embed_dim)
        selfies_encoder.load_state_dict(selfies_ae_checkpoint["encoder_model"])

        prot_ae_path = "Models/prot_autoencoder.pth"
        prot_ae_checkpoint = torch.load(prot_ae_path)
        prot_embed_dim = prot_ae_checkpoint["embedding_dim"]
        prot_chunk_len = prot_ae_checkpoint["chunk_length"]
        prot_encoder = ProtEncoder(len(prot_alphabet), prot_chunk_len=prot_chunk_len, hidden_size=prot_embed_dim)
        prot_encoder.load_state_dict(prot_ae_checkpoint["encoder_model"])

        assert max_selfies_len % selfies_chunk_len == 0
        assert max_prot_len % prot_chunk_len == 0

        self.total_selfies_dim = (max_selfies_len // selfies_chunk_len) * selfies_embed_dim
        self.total_prot_dim = (max_prot_len // prot_chunk_len) * prot_embed_dim

        prot_seq_df = pd.read_csv("Data/prot_ids.tsv", sep="\t")
        prot_seq_dict = dict(zip(list(prot_seq_df["Entry"]), list(prot_seq_df["Sequence"])))

        print("Processing Data Points")
        self.data_points = []
        prot_id_counts = {}
        self.prot_id_to_embed_dict = {}
        self.smiles_to_embed_dict = {}

        with open("Data/BindingDB_All.tsv", "r") as f:
            headers = f.readline()
            column_names = headers.split("\t")
            smiles_idx = column_names.index("Ligand SMILES")
            prot_idx = column_names.index("UniProt (SwissProt) Primary ID of Target Chain 1")
            IC50_idx = column_names.index("IC50 (nM)")

            line_number = 0
            for line in f:
                line_data = line.split("\t")
                smiles = line_data[smiles_idx]
                prot_id = line_data[prot_idx]
                IC50 = line_data[IC50_idx]

                if line_number % 1000 == 0:
                    print(f"Line Number: {line_number}, Processed {len(self.data_points)} Data Points")
                line_number += 1

                if line_number > max_lines:
                    break

                if prot_id in prot_id_counts and prot_id_counts[prot_id] >= max_entries_per_prot:
                    # print("Protein already past max entries")
                    continue

                try:
                    selfies = list(sf.split_selfies(sf.encoder(smiles)))
                    prot_seq = prot_seq_dict[prot_id]
                    pIC50 = -1 * np.log10(float(IC50))
                    
                    for metric in [pIC50]:
                        if np.isinf(metric) or np.isnan(metric):
                            # print("pIC50 is NaN or Inf")
                            continue

                    if pIC50 < -4.5 or pIC50 > -0.5:
                        # print(f"IC50 is too large or too small: {pIC50}")
                        continue

                    pIC50 = (pIC50 + 4.5) / 4.0

                except Exception as e:
                    # print(e)
                    continue

                if len(selfies) > max_selfies_len:
                    # print("Selfies is too long")
                    continue

                if len(prot_seq) > max_prot_len:
                    # print("Protein is too long")
                    continue
                
                try:
                    with torch.no_grad():
                        if smiles not in self.smiles_to_embed_dict:
                            self.smiles_to_embed_dict[smiles] = smiles_to_embedding(smiles, selfies_alphabet, selfies_encoder, selfies_chunk_len, max_selfies_len).flatten()
                        if prot_id not in self.prot_id_to_embed_dict:
                            self.prot_id_to_embed_dict[prot_id] = prot_to_embedding(prot_seq, prot_alphabet, prot_encoder, prot_chunk_len, max_prot_len).flatten()
                except Exception as e:
                    # print(e)
                    continue

                if prot_id not in prot_id_counts:
                    prot_id_counts[prot_id] = 1
                else:
                    prot_id_counts[prot_id] += 1
                
                self.data_points.append([prot_id, smiles, pIC50])     
        
        self.data_points.sort(key=lambda x: x[1])
        print("Finished Processing Data")

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, index):
        data_points = self.data_points[index]
        prot_id = data_points[0]
        smiles = data_points[1]
        pIC50 = data_points[2]

        prot_tensor = torch.tensor(self.prot_id_to_embed_dict[prot_id], dtype=torch.float32)
        selfies_tensor = torch.tensor(self.smiles_to_embed_dict[smiles], dtype=torch.float32)
        pIC50_tensor = torch.tensor([pIC50], dtype=torch.float32)

        return prot_tensor, selfies_tensor, pIC50_tensor
    
    def get_selfies_embed_dim(self):
        return self.total_selfies_dim
    
    def get_prot_embed_dim(self):
        return self.total_prot_dim

class ProtLigandModel(nn.Module):
    def __init__(self, prot_embed_dim, selfies_embed_dim, hidden_size=128):
        super().__init__()
        self.input_layer = nn.Linear(prot_embed_dim + selfies_embed_dim, 512)

        self.fc1 = nn.Linear(512, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.LayerNorm(hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

        self.activation = nn.GELU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, prot_tensor, selfies_tensor):
        input_tensor = torch.cat([prot_tensor, selfies_tensor], dim=-1)
        x = self.input_layer(input_tensor)
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.activation(self.bn2(self.fc2(x)))
        # x = self.activation(self.bn3(self.fc3(x)))
        x = self.output_layer(x)

        return x

def main():
    train_test_split = 0.1
    batch_size = 128
    hidden_size = 128
    max_selfies_len=100
    max_prot_len=1000
    max_entries_per_prot=100
    max_lines=200000
    lr = 1e-4
    lr_limit = 1e-6
    weight_decay = 1e-3
    noise_std = 0.0
    model_savefile = "Models/prot_ligand_model.pth"
    set_seeds(1234)

    dataset = ProtLigandDataset(max_prot_len, max_selfies_len, max_entries_per_prot, max_lines)
    train_size = int(len(dataset) * (1 - train_test_split))
    indices = list(range(len(dataset)))
    # random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = nn.L1Loss()

    model = ProtLigandModel(dataset.get_prot_embed_dim(), dataset.get_selfies_embed_dim(), hidden_size=hidden_size)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, threshold=1e-4)

    if os.path.exists(model_savefile):
        model_checkpoint = torch.load(model_savefile)
        model.load_state_dict(model_checkpoint["model_state_dict"])
    
    current_lr = lr
    while current_lr > lr_limit:
        model.train()
        train_loss = 0.0
        batch = 0
        for prot_tensor, selfies_tensor, ic50_tensor in train_loader:
            optimizer.zero_grad()
            selfies_tensor_noisy = selfies_tensor + noise_std * torch.randn_like(selfies_tensor)
            prot_tensor_noisy = prot_tensor + noise_std * torch.randn_like(prot_tensor)
            outputs = model(prot_tensor_noisy, selfies_tensor_noisy)
            loss = criterion(outputs, ic50_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

            train_loss += loss.item()
            batch += 1
        
        print(f"Training Loss = {train_loss / batch}")
        torch.save({"model_state_dict": model.state_dict(),
                    "hidden_size": hidden_size,
                    "max_selfies_len": max_selfies_len,
                    "max_prot_len": max_prot_len},
                    model_savefile)

        model.eval()
        test_loss = 0.0
        batch = 0
        total_outputs = torch.tensor([], dtype=torch.float32)
        total_ic50 = torch.tensor([], dtype=torch.float32)
        for prot_tensor, selfies_tensor, ic50_tensor in test_loader:
            outputs = model(prot_tensor, selfies_tensor)
            loss = criterion(outputs, ic50_tensor)

            total_outputs = torch.cat([total_outputs, outputs], dim=0)
            total_ic50 = torch.cat([total_ic50, ic50_tensor], dim=0)

            test_loss += loss.item()
            batch += 1
        
        total_outputs = total_outputs.detach().cpu().numpy().flatten()
        total_ic50 = total_ic50.detach().cpu().numpy().flatten()
        scheduler.step(test_loss / batch)
        print(f"Testing Loss = {test_loss / batch}")
        print(f"Pearson Correlation = {pearsonr(total_outputs, total_ic50)[0]}")
        print(f"Spearman Correlation = {spearmanr(total_outputs, total_ic50)[0]}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR = {current_lr}")

if __name__=="__main__":
    main()