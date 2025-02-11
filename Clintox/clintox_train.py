from torch_geometric.loader import DataLoader
from torch_geometric.graphgym import init_weights
from Clintox.clintox_dataset import ClintoxDataset
import torch
from Model.ismorphism import GCNEncoder
from Model.model_classification import MoleculePropertyClassifier
from metrics import classification_binary_metrics
from torch.utils.data import ConcatDataset
import torch.multiprocessing as tmp
from torch import nn
from dotenv import load_dotenv
import os
import wandb
import gc


def train_epoch():
    epoch_loss = 0
    epoch_acc = 0
    epoch_prec = 0
    epoch_rec = 0
    epoch_auc = 0
    epoch_f1 = 0
    epoch_auc = 0

    for step, graphs in enumerate(train_loader):
        logits, predictions = model(
            graphs.x, edges=graphs.edge_index, batch=graphs.batch)

        target_col = graphs.y.view(graphs.y.size(0), 1)
        # Train the model
        model.zero_grad()
        loss_function = nn.BCEWithLogitsLoss()
        loss = loss_function(logits, target_col.float())

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        acc, f1, prec, rec, auc = classification_binary_metrics(
            predictions, target_col.int())
        epoch_acc += acc.item()
        epoch_prec += prec.item()
        epoch_f1 += f1.item()
        epoch_rec += rec.item()
        epoch_auc += auc.item()

        del graphs, predictions, logits
        gc.collect()
    return epoch_loss/(step+1), epoch_acc/(step+1), epoch_prec/(step+1), epoch_rec/(step+1), epoch_f1/(step+1), \
        epoch_auc/(step+1)


def test_epoch():
    epoch_loss = 0
    epoch_acc = 0
    epoch_prec = 0
    epoch_rec = 0
    epoch_auc = 0
    epoch_f1 = 0
    epoch_auc = 0

    for step, graphs in enumerate(train_loader):
        logits, predictions = model(
            graphs.x, edges=graphs.edge_index, batch=graphs.batch)
        target_col = graphs.y.view(graphs.y.size(0), 1)

        loss_function = nn.BCEWithLogitsLoss()
        loss = loss_function(logits, target_col.float())

        epoch_loss += loss.item()

        acc, f1, prec, rec, auc = classification_binary_metrics(
            predictions, target_col.int())
        epoch_acc += acc.item()
        epoch_prec += prec.item()
        epoch_f1 += f1.item()
        epoch_rec += rec.item()
        epoch_auc += auc.item()

        del graphs, predictions, logits
        gc.collect()

    return epoch_loss/(step+1), epoch_acc/(step+1), epoch_prec/(step+1), epoch_rec/(step+1), epoch_f1/(step+1), \
        epoch_auc/(step+1)


def training_loop():
    for epoch in range(NUM_EPOCHS):
        model.train(True)
        train_loss, train_acc, train_prec, train_rec, train_f1, train_auc = train_epoch()

        model.eval()

        with torch.no_grad():
            test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = test_epoch()

            print(f"Epoch: {epoch+1}")
            print("----------Train Metrics------------")
            print(f"Train Loss: {train_loss}")
            print(f"Train Accuracy: {train_acc}")
            print(f"Train Precision: {train_prec}")
            print(f"Train Recall: {train_rec}")
            print(f"Train F1: {train_f1}")
            print(f"Train AUC: {train_auc}")
            print("------------Test Metrics-------------")
            print(f"Test Loss: {test_loss}")
            print(f"Test Accuracy: {test_acc}")
            print(f"Test Precision: {test_prec}")
            print(f"Test Recall: {test_rec}")
            print(f"Test F1: {test_f1}")
            print(f"Test AUC: {test_auc}")

            wandb.log({
                "Train Loss": train_loss,
                "Train Accuracy": train_acc,
                "Train Precision": train_prec,
                "Train Recall": train_rec,
                "Train F1": train_f1,
                "Train AUC": train_auc,
                "Test Loss": test_loss,
                "Test Accuracy": test_acc,
                "Test Precision": test_prec,
                "Test Recall": test_rec,
                "Test F1": test_f1,
                "Test AUC": test_auc
            })

            if (epoch+1) % 10 == 0:
                weights_path = f"Clintox/weights/run_1/model_{epoch+1}.pth"
                torch.save(model.state_dict(), weights_path)


if __name__ == '__main__':
    tmp.set_sharing_strategy('file_system')
    torch.autograd.set_detect_anomaly(True)
    load_dotenv('.env')

    train_folds = ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5', 'Fold6']
    test_folds = ['Fold7', 'Fold8']

    train_set1 = ClintoxDataset(
        fold_key=train_folds[0], root=os.getenv("graph_files")+"/Fold1"+"/data/", start=0, stop=6)
    train_set2 = ClintoxDataset(fold_key=train_folds[1], root=os.getenv("graph_files")+"/Fold2/"
                                + "/data/", start=185, stop=301)
    train_set3 = ClintoxDataset(fold_key=train_folds[2], root=os.getenv("graph_files")+"/Fold3/"
                                + "/data/", start=370, stop=554)
    train_set4 = ClintoxDataset(fold_key=train_folds[3], root=os.getenv("graph_files")+"/Fold4/"
                                + "/data/", start=555, stop=739)
    train_set5 = ClintoxDataset(fold_key=train_folds[4], root=os.getenv("graph_files")+"/Fold5/"
                                + "/data/", start=740, stop=924)
    train_set6 = ClintoxDataset(fold_key=train_folds[5], root=os.getenv("graph_files")+"/Fold6/"
                                + "/data/", start=925, stop=1109)

    test_set1 = ClintoxDataset(fold_key=test_folds[0], root=os.getenv("graph_files")+"/Fold7/"
                               + "/data/", start=1110, stop=1217)
    test_set2 = ClintoxDataset(fold_key=test_folds[1], root=os.getenv(
        "graph_files")+"/Fold8"+"/data/", start=1295, stop=1479)

    train_set = ConcatDataset(
        [train_set1, train_set2, train_set3, train_set4, train_set5, train_set6])
    test_set = ConcatDataset([test_set1, test_set2])

    params = {
        'batch_size': 128,
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

    model = MoleculePropertyClassifier(num_labels=1, encoder=r_enc)

    NUM_EPOCHS = 1000
    LR = 0.001
    BETAS = (0.9, 0.999)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, betas=BETAS)

    training_loop()
