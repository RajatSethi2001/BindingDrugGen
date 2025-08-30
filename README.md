# Proteomic Reinforcement Learning Drug Generator

This is an independent project designed by Rajat Sethi. Using RL, this system creates treatments that target
specific pathogens. There are four primary components required:

- Selfies Autoencoder
    - Converts SMILES into SELFIES into embeddings
    - Splits SELFIES into chunks to make embeddings more accurate.
    - Includes decoder to convert embedding back into SELFIES.

- Protein Autoencoder
    - Converts proteins into embeddings.
    - Splits proteins into chunks to make embeddings more accurate.
    - Includes decoder to convert embedding back into protein.

- Protein-Ligand Binding Model
    - Uses BindingDB data to estimate the binding affinity of a ligand to protein.
    - Affinity scored using IC50, normalized by -log10 and then scaled from [-4.5, -0.5] to [0, 1].
    - Higher value represents stronger affinity.
    - Input is the protein's embedding and ligand's embedding.

- Drug Generator
    - Uses stable-baselines3's PPO framework to generate drugs.
    - Observation space is the protein embedding.
    - Action space is the selfies embedding.
    - Molecule rewarded on estimated binding affinity, QED score, and SA Score.
    - Training will dynamically plot reward (with curve smoothing) and print the "highest" reward per protein.

## Download Prerequisites

Install Python libraries with pip:
```
pip install -r requirements.txt
```

Download BindingDB_All.tsv, which contains information about protein-ligand binding.
https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp

Download CHEMBL 35.
https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/

## Training Prerequisites

Train the models in the following order. They will automatically stop training at a certain point.
Run their respective python scripts, no command-line arguments at this point.

1. Selfies Autoencoder
2. Protein Autoencoder
3. Protein Ligand Model
4. Drug Generator