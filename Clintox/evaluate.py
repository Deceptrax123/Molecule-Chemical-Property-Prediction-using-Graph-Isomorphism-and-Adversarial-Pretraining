import torch
from Clintox.clintox_dataset import ClintoxDataset
from torch_geometric.loader import DataLoader
from torch_geometric.graphgym import init_weights
from metrics import classification_binary_metrics
from Model.model_classification import MoleculePropertyClassifier
from Model.ismorphism import GCNEncoder
import torch.multiprocessing as tmp
from dotenv import load_dotenv
import os
import numpy as np
import gc


def evaluate():
    epoch_acc = 0
    epoch_prec = 0
    epoch_rec = 0
    epoch_auc = 0
    epoch_f1 = 0
    epoch_auc = 0

    for step, graphs in enumerate(test_loader):
        logits, predictions = model(
            graphs.x, edges=graphs.edge_index, batch=graphs.batch)

        acc, f1, prec, rec, auc = classification_binary_metrics(
            predictions, graphs.y)
        epoch_acc += acc.item()
        epoch_prec += prec.item()
        epoch_f1 += f1.item()
        epoch_rec += rec.item()
        epoch_auc += auc.item()

        del graphs, predictions, logits
        gc.collect()

    return epoch_acc/(step+1), epoch_prec/(step+1), epoch_rec/(step+1), epoch_f1/(step+1), \
        epoch_auc/(step+1)


if __name__ == '__main__':
    load_dotenv('.env')

    params = {
        'batch_size': 16,
        'shuffle': True
    }

    test_set = ClintoxDataset(fold_key='val', root=os.getenv(
        "graph_files")+"/val"+"/data/")
    test_loader = DataLoader(test_set, **params)

    r_enc = GCNEncoder()

    model = MoleculePropertyClassifier(num_labels=1, encoder=r_enc)
    # Trained weights are loaded here
    model.load_state_dict("")

    model.eval()

    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = evaluate()

    print("------------Test Metrics-------------")
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_acc}")
    print(f"Test Precision: {test_prec}")
    print(f"Test Recall: {test_rec}")
    print(f"Test F1: {test_f1}")
    print(f"Test AUC: {test_auc}")
