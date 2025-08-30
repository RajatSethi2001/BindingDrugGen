import numpy as np
import pandas as pd
import random
from utils import set_seeds

def main():
    prot_ids = pd.read_csv("Data/BindingDB_All.tsv", sep="\t", usecols=['UniProt (SwissProt) Primary ID of Target Chain 1'])['UniProt (SwissProt) Primary ID of Target Chain 1']
    prot_ids = prot_ids[~pd.isna(prot_ids)]
    data = set(prot_ids)
    print(*data, sep=",")

if __name__=="__main__":
    main()