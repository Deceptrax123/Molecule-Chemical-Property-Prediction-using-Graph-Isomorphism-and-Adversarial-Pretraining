from torch_geometric.loader import DataLoader
from torch_geometric.graphgym import init_weights
from Liphophilicity.lipho_dataset import LiphophilicityDataset
import torch
from Model.ismorphism import GCNEncoder
from Model.model_regression import MoleculePropertyRegression
from metrics import classification_binary_metrics
from torch.utils.data import ConcatDataset
import torch.multiprocessing as tmp
from torch import nn
import math
from dotenv import load_dotenv
import os
import wandb
import gc


def train_epoch():
    epoch_loss = 0

    for step, graphs in enumerate(train_loader):
        predictions = model(
            graphs.x, edges=graphs.edge_index, batch=graphs.batch)

        # Train the model
        model.zero_grad()
        target_col = graphs.y.view(graphs.y.size(0), 1)
        loss = loss_function(predictions, target_col.float())

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        del graphs, predictions
        gc.collect()

    return epoch_loss/(step+1)


def test_epoch():
    epoch_loss = 0

    for step, graphs in enumerate(train_loader):
        predictions = model(
            graphs.x, edges=graphs.edge_index, batch=graphs.batch)
        target_col = graphs.y.view(graphs.y.size(0), 1)
        loss = loss_function(predictions, target_col.float())

        epoch_loss += loss.item()

        del graphs, predictions
        gc.collect()

    return epoch_loss/(step+1)


def training_loop():
    for epoch in range(NUM_EPOCHS):
        model.train(True)
        train_loss = train_epoch()

        model.eval()

        with torch.no_grad():
            test_loss = test_epoch()

            print(f"Epoch: {epoch+1}")
            print("----------Train Metrics------------")
            print(f"Train Loss: {train_loss}")
            print(f"Train RMSE: {math.sqrt(train_loss)}")
            print("------------Test Metrics-------------")
            print(f"Test Loss: {test_loss}")
            print(f"Test RMSE: {math.sqrt(test_loss)}")

            wandb.log({
                "Train Loss": train_loss,
                "Test Loss": test_loss,
                "Train Root Mean Square Error": math.sqrt(train_loss),
                "Test Root Mean Square Error": math.sqrt(test_loss)
            })

            if (epoch+1) % 10 == 0:
                weights_path = f"Liphophilicity/weights/run_1/model_{
                    epoch+1}.pth"
                torch.save(model.state_dict(), weights_path)


if __name__ == '__main__':
    tmp.set_sharing_strategy('file_system')
    torch.autograd.set_detect_anomaly(True)
    load_dotenv('.env')

    train_folds = ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5', 'Fold6']
    test_folds = ['Fold7', 'Fold8']

    train_set1 = LiphophilicityDataset(
        fold_key=train_folds[0], root=os.getenv("graph_files")+"/Fold1"+"/data/", start=0)
    train_set2 = LiphophilicityDataset(fold_key=train_folds[1], root=os.getenv("graph_files")+"/Fold2/"
                                       + "/data/", start=525)
    train_set3 = LiphophilicityDataset(fold_key=train_folds[2], root=os.getenv("graph_files")+"/Fold3/"
                                       + "/data/", start=1050)
    train_set4 = LiphophilicityDataset(fold_key=train_folds[3], root=os.getenv("graph_files")+"/Fold4/"
                                       + "/data/", start=1575)
    train_set5 = LiphophilicityDataset(fold_key=train_folds[4], root=os.getenv("graph_files")+"/Fold5/"
                                       + "/data/", start=2100)
    train_set6 = LiphophilicityDataset(fold_key=train_folds[5], root=os.getenv("graph_files")+"/Fold6/"
                                       + "/data/", start=2625)

    test_set1 = LiphophilicityDataset(fold_key=test_folds[0], root=os.getenv("graph_files")+"/Fold7/"
                                      + "/data/", start=3150)
    test_set2 = LiphophilicityDataset(fold_key=test_folds[1], root=os.getenv(
        "graph_files")+"/Fold8"+"/data/", start=3675)

    train_set = ConcatDataset(
        [train_set1, train_set2, train_set3, train_set4, train_set5, train_set6])
    test_set = ConcatDataset([test_set1, test_set2])

    params = {
        'batch_size': 64,
        'shuffle': True,
        'num_workers': 0
    }

    train_loader = DataLoader(train_set, **params)
    test_loader = DataLoader(test_set, **params)

    wandb.init(
        project="Molecular Property Prediction",
        config={
            "Method": "Graph Isomorphism and Contrastive",
        })

    r_enc = GCNEncoder()
    # Load pre-trained weights here
    extractor = os.getenv("zinc_weights")
    r_enc.load_state_dict(torch.load(extractor))

    model = MoleculePropertyRegression(num_labels=1, encoder=r_enc)
    loss_function = nn.MSELoss()

    NUM_EPOCHS = 1000
    LR = 2e-4
    BETAS = (0.9, 0.999)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, betas=BETAS)

    train_steps = (len(train_set)+params['batch_size']-1)//params['batch_size']
    test_steps = (len(test_set)+params['batch_size']-1)//params['batch_size']

    training_loop()
