import torch
from HIV.hiv_dataset import HIVDataset
from torch_geometric.loader import DataLoader
from Model.model_classification import MoleculePropertyClassifier
from Model.ismorphism import GCNEncoder
import torch.multiprocessing as tmp
from dotenv import load_dotenv
from torch import nn
import os
import numpy as np
import gc


def evaluate():
    epoch_mse = 0

    for step, graphs in enumerate(test_loader):
        logits, predictions = model(
            graphs.x, edges=graphs.edge_index, batch=graphs.batch)

        epoch_mse += mse(graphs.y, predictions).item()

        del graphs, predictions, logits
        gc.collect()

    return epoch_mse/(step+1)


if __name__ == '__main__':
    load_dotenv('.env')

    params = {
        'batch_size': 16,
        'shuffle': True
    }

    test_set = HIVDataset(fold_key='val', root=os.getenv(
        "graph_files")+"/val"+"/data/")
    test_loader = DataLoader(test_set, **params)

    r_enc = GCNEncoder()

    model = MoleculePropertyClassifier(num_labels=1, encoder=r_enc)
    # Trained weights are loaded here
    model.load_state_dict("")

    model.eval()
    mse = nn.MSELoss()

    mse_loss = evaluate()
    rmse = torch.sqrt(torch.tensor(mse_loss))

    print("------------Test Metrics-------------")
    print(f"Test Loss: {mse_loss}")
    print(f"Root Mean Square Error: {rmse.item()}")
