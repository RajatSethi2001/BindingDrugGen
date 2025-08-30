import numpy as np
import os
import pandas as pd
import random
import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, Subset
from utils import one_hot_encode, one_hot_decode, set_seeds

class ProtEncoder(nn.Module):
    def __init__(self, prot_alphabet_len, prot_chunk_len=20, hidden_size=64):
        super().__init__()
        self.input_size = prot_alphabet_len * prot_chunk_len
        self.input_layer = nn.Linear(self.input_size, hidden_size)

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.LayerNorm(hidden_size)
        
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.GELU()

    def forward(self, prot_one_hot):
        x = prot_one_hot.reshape(-1, self.input_size)
        x = self.input_layer(x)
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.activation(self.bn2(self.fc2(x)))
        output = self.sigmoid(self.output_layer(x))
        return output
    
class ProtDecoder(nn.Module):
    def __init__(self, prot_alphabet_len, prot_chunk_len=20, hidden_size=64):
        super().__init__()
        self.prot_alphabet_len = prot_alphabet_len
        self.prot_chunk_len = prot_chunk_len
        self.output_size = prot_alphabet_len * prot_chunk_len
        
        self.input_layer = nn.Linear(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.LayerNorm(hidden_size)
        
        self.output_layer = nn.Linear(hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.GELU()

    def forward(self, selfies_embedding):
        x = self.input_layer(selfies_embedding)
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.activation(self.bn3(self.fc3(x)))
        output = self.sigmoid(self.output_layer(x))
        output = output.view(-1, self.prot_chunk_len, self.prot_alphabet_len)
        return output

class ProtDataset(Dataset):
    def __init__(self, prot_set, prot_alphabet, prot_chunk_len=20):
        self.prot_one_hot_list = []
        for prot in prot_set:
            prot_tokens = list(prot)
            prot_tokens += ["_" for _ in range(prot_chunk_len - len(prot_tokens) % prot_chunk_len)]
            for prot_idx in range(0, len(prot_tokens), prot_chunk_len):
                prot_chunk_tokens = prot_tokens[prot_idx:prot_idx+prot_chunk_len]
                try:
                    prot_one_hot = one_hot_encode(prot_chunk_tokens, prot_alphabet)
                except:
                    continue
                self.prot_one_hot_list.append(torch.tensor(prot_one_hot, dtype=torch.float32))

    def __len__(self):
        return len(self.prot_one_hot_list)

    def __getitem__(self, index):
        return self.prot_one_hot_list[index]

def main():
    set_seeds(2222)
    train_test_split = 0.2
    prot_chunk_len = 25
    prot_file = "Data/prot_seqs.tsv"
    nrows=20000
    save_dir = "Models"

    batch_size = 128
    input_noise = 0.0

    hidden_size = 150
    lr = 1e-4
    weight_decay = 1e-3
    lr_limit = 1e-6

    prot_df = pd.read_csv(prot_file, sep="\t", nrows=nrows)
    prot_set = set(prot_df["Sequence"].to_list())

    with open("Data/prot_alphabet.txt", "r") as f:
        prot_alphabet = f.read().splitlines()
    
    dataset = ProtDataset(prot_set, prot_alphabet, prot_chunk_len=prot_chunk_len)
    train_size = int(len(dataset) * (1 - train_test_split))
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = nn.BCELoss()
    encoder = ProtEncoder(len(prot_alphabet),
                            prot_chunk_len=prot_chunk_len,
                            hidden_size=hidden_size)
    
    decoder = ProtDecoder(len(prot_alphabet),
                             prot_chunk_len=prot_chunk_len,
                             hidden_size=hidden_size)

    encoder_optim = optim.AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    decoder_optim = optim.AdamW(decoder.parameters(), lr=lr, weight_decay=weight_decay)

    if os.path.exists(f"{save_dir}/prot_autoencoder.pth"):
        checkpoint = torch.load(f"{save_dir}/prot_autoencoder.pth")
        encoder.load_state_dict(checkpoint["encoder_model"])
        decoder.load_state_dict(checkpoint["decoder_model"])
    
    encoder_lr = lr
    decoder_lr = lr
    encode_scheduler = ReduceLROnPlateau(encoder_optim, mode='min', factor=0.5, patience=3, threshold=1e-3)
    decode_scheduler = ReduceLROnPlateau(decoder_optim, mode='min', factor=0.5, patience=3, threshold=1e-3)

    while encoder_lr > lr_limit:
        encoder.train()
        decoder.train()
        train_loss = 0.0
        batch_idx = 0
        for prot_one_hot in train_loader:
            noisy_prot_one_hot = prot_one_hot + input_noise * torch.randn_like(prot_one_hot)
            encoder_optim.zero_grad()
            decoder_optim.zero_grad()
            prot_encoding = encoder(noisy_prot_one_hot)
            prot_decoding = decoder(prot_encoding)
            
            loss = criterion(prot_decoding, prot_one_hot)
            loss.backward()
            train_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            encoder_optim.step()
            decoder_optim.step()

            batch_idx += 1
            # if batch_idx % 5 == 0:
            #     print(f"Training Batch Loss = {loss.item()}")

        print("Saving Encoder and Decoder")
        checkpoint = {
            "encoder_model": encoder.state_dict(),
            "decoder_model": decoder.state_dict(),
            "embedding_dim": hidden_size,
            "chunk_length": prot_chunk_len
        }
        torch.save(checkpoint, f"{save_dir}/prot_autoencoder.pth")
        
        encoder.eval()
        decoder.eval()
        test_loss = 0.0
        for prot_one_hot in test_loader:
            prot_encoding = encoder(prot_one_hot)
            prot_decoding = decoder(prot_encoding)

            loss = criterion(prot_decoding, prot_one_hot)
            test_loss += loss.item()

            prot_probs_np = prot_decoding.detach().cpu().numpy()
        
        correct_prots = 0
        total_prots = 0
        correct_prot_tokens = 0
        total_prot_tokens = 0
        for batch_idx in range(batch_size):
            prot_one_hot_og = prot_one_hot[batch_idx].detach().cpu().numpy()
            prot_og_tokens = one_hot_decode(prot_one_hot_og, prot_alphabet)
            prot_og_tokens_clean = [token for token in prot_og_tokens if token != "_"]
            prot_og = "".join(prot_og_tokens_clean)
            
            prot_one_hot_ae = prot_probs_np[batch_idx]
            prot_ae_tokens = one_hot_decode(prot_one_hot_ae, prot_alphabet)
            prot_ae_tokens_clean = [token for token in prot_ae_tokens if token != "_"]
            prot_ae = "".join(prot_ae_tokens_clean)

            if prot_og == prot_ae:
                correct_prots += 1
            total_prots += 1

            for token_idx in range(len(prot_og_tokens)):
                if prot_og_tokens[token_idx] == prot_ae_tokens[token_idx]:
                    correct_prot_tokens += 1
                total_prot_tokens += 1

            print(f"Protein Before: {prot_og}")
            print(f"Protein After:  {prot_ae}")
            print()
            
        print(f"Full Prot Accuracy = {correct_prots / total_prots}")
        print(f"Token-Wide Prot Accuracy = {correct_prot_tokens / total_prot_tokens}")
        test_loss = test_loss / (len(test_dataset) // batch_size)
        print(f"Training Loss = {train_loss / (len(train_dataset) // batch_size)}")
        print(f"Testing Loss = {test_loss}")
        encode_scheduler.step(test_loss)
        decode_scheduler.step(test_loss)

        encoder_lr = encoder_optim.param_groups[0]['lr']
        decoder_lr = decoder_optim.param_groups[0]['lr']
        print(f"Encoder LR = {encoder_lr}, Decoder LR = {decoder_lr}")

if __name__=="__main__":
    main()