import torch
from Liphophilicity.lipho_dataset import LiphophilicityDataset
from torch_geometric.loader import DataLoader
from Model.model_classification import MoleculePropertyClassifier
from Model.ismorphism import GCNEncoder
import torch.multiprocessing as tmp
from dotenv import load_dotenv
from torch import nn
import os
import math
import gc


def evaluate():
    epoch_mse = 0

    for step, graphs in enumerate(test_loader):
        logits, predictions = model(
            graphs.x, edges=graphs.edge_index, batch=graphs.batch)
        target_col = graphs.y.view(graphs.y.size(0), 1)
        epoch_mse += mse(predictions, target_col.float()).item()

        del graphs, predictions, logits
        gc.collect()

    return epoch_mse/(step+1)


if __name__ == '__main__':
    load_dotenv('.env')

    params = {
        'batch_size': 64,
        'shuffle': True
    }

    test_set = LiphophilicityDataset(fold_key='Fold8', root=os.getenv(
        "graph_files")+"/Fold8"+"/data/", start=3675)
    test_loader = DataLoader(test_set, **params)

    r_enc = GCNEncoder()

    model = MoleculePropertyClassifier(num_labels=1, encoder=r_enc)
    # Trained weights are loaded here
    model.load_state_dict(torch.load(
        "Liphophilicity/weights/run_1/model_1000.pth"))

    model.eval()
    mse = nn.MSELoss()

    mse_loss = evaluate()
    rmse = math.sqrt(mse_loss)

    print("------------Test Metrics-------------")
    print(f"Test Loss: {mse_loss}")
    print(f"Root Mean Square Error: {rmse}")
