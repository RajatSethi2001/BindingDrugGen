import numpy as np
import random
import selfies as sf
import torch

def one_hot_encode(tokens, alphabet):
    vocab_dict = {token: i for i, token in enumerate(alphabet)}
    indices = [vocab_dict[token] for token in tokens]
    one_hot = np.eye(len(vocab_dict))[indices]
    return one_hot

def one_hot_decode(one_hot, alphabet):
    tokens = []
    for vector in one_hot:
        index = np.argmax(vector)
        token = alphabet[index]
        tokens.append(token)
    return tokens

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def smiles_to_embedding(smiles, selfies_alphabet, encoder, selfies_chunk_len, max_selfies_len):
    selfies = sf.encoder(smiles)
    selfies_tokens = list(sf.split_selfies(selfies))
    selfies_tokens += ["[SKIP]" for _ in range(max_selfies_len - len(selfies_tokens))]
    selfies_one_hot_list = []
    for token_idx in range(0, len(selfies_tokens), selfies_chunk_len):
        chunk_tokens = selfies_tokens[token_idx:token_idx + selfies_chunk_len]
        chunk_one_hot = one_hot_encode(chunk_tokens, selfies_alphabet)
        selfies_one_hot_list.append(chunk_one_hot)
    
    selfies_one_hot_tensor = torch.tensor(selfies_one_hot_list, dtype=torch.float32)
    selfies_embeddings = encoder(selfies_one_hot_tensor)
    return selfies_embeddings.detach().cpu().numpy()

def embedding_to_smiles(embedding_list, selfies_alphabet, decoder):
    embedding_tensor = torch.tensor(embedding_list, dtype=torch.float32)
    selfies_one_hot = decoder(embedding_tensor).detach().cpu().numpy()
    selfies = ""
    for chunk_one_hot in selfies_one_hot:
        chunk_tokens = one_hot_decode(chunk_one_hot, selfies_alphabet)
        chunk_tokens_clean = [token for token in chunk_tokens if token != "[SKIP]"]
        selfies += "".join(chunk_tokens_clean)
    smiles = sf.decoder(selfies)
    return smiles

def prot_to_embedding(prot_seq, prot_alphabet, encoder, prot_chunk_len, max_prot_len):
    prot_tokens = list(prot_seq)
    prot_tokens += ["_" for _ in range(max_prot_len - len(prot_tokens))]
    prot_one_hot_list = []
    for token_idx in range(0, len(prot_tokens), prot_chunk_len):
        chunk_tokens = prot_tokens[token_idx:token_idx + prot_chunk_len]
        chunk_one_hot = one_hot_encode(chunk_tokens, prot_alphabet)
        prot_one_hot_list.append(chunk_one_hot)
    
    prot_one_hot_tensor = torch.tensor(prot_one_hot_list, dtype=torch.float32)
    prot_embeddings = encoder(prot_one_hot_tensor)
    return prot_embeddings.detach().cpu().numpy()

def embedding_to_prot(embedding_list, prot_alphabet, decoder):
    embedding_tensor = torch.tensor(embedding_list, dtype=torch.float32)
    prot_one_hot = decoder(embedding_tensor).detach().cpu().numpy()
    prot = ""
    for chunk_one_hot in prot_one_hot:
        chunk_tokens = one_hot_decode(chunk_one_hot, prot_alphabet)
        chunk_tokens_clean = [token for token in chunk_tokens if token != "_"]
        prot += "".join(chunk_tokens_clean)
    return prot