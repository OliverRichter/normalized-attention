import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from tqdm import tqdm
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score

from attention_conv import TransformerConv

path = osp.join(osp.dirname(osp.realpath(__file__)), 'pytorch_geometric', 'data', 'PPI')
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


def create_and_run_model(learning_rate, model, model_dimension, epochs, heads, layers, **unused_kwargs):

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.embedding = Linear(train_dataset.num_features, model_dimension)
            self.convs = ModuleList()
            for layer in range(layers):
                self.convs.append(TransformerConv(model_dimension, heads, model))
            self.last_layer = Linear(model_dimension, train_dataset.num_classes)

        def forward(self, x, edge_index):
            x = self.embedding(x)
            for conv in self.convs:
                x = conv(x, edge_index)
            x = self.last_layer(x)
            return x

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_net = Net().to(device)
    print(model_net)
    loss_op = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model_net.parameters(), lr=learning_rate)  # lr = 0.005
    total_params = sum(p.numel() for p in model_net.parameters() if p.requires_grad)
    print('Parameters: ', total_params)

    def train(epoch):
        model_net.train()
        pbar = tqdm(total=len(train_loader.dataset))
        pbar.set_description(f'Epoch {epoch:02d}')

        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = loss_op(model_net(data.x, data.edge_index), data.y)
            total_loss += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
            pbar.update(data.num_graphs)
        pbar.close()
        return total_loss / len(train_loader.dataset)

    @torch.no_grad()
    def test(loader):
        model_net.eval()

        ys, preds = [], []
        for data in loader:
            ys.append(data.y)
            out = model_net(data.x.to(device), data.edge_index.to(device))
            preds.append((out > 0).float().cpu())

        y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
        return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

    class History:
        def __init__(self):
            self.history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

    history = History()
    for epoch in range(epochs):
        loss = train(epoch)
        train_f1 = test(train_loader)
        val_f1 = test(val_loader)
        test_f1 = test(test_loader)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, '
              f'Train: {train_f1:.4f}, Val: {val_f1:.4f}, Test: {test_f1:.4f}', flush=True)
        history.history['accuracy'].append(train_f1)
        history.history['loss'].append(loss)
        history.history['val_accuracy'].append(val_f1)
        history.history['val_loss'].append(test_f1)
    del model_net
    return history
