import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

LEARNING_RATE = 0.001
EPOCH = 100
BATCH_SIZE = 64

class MyDataset(Dataset):
    def __init__(self, data_X: np, data_Y: np, transform):
        self.data = data_X
        self.labels = data_Y
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img = self.transform(self.data[index])
        label = self.labels[index]
        return img, label
    

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # input (3, 32, 32)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3) # (16, 30, 30)
        self.pool1 = nn.MaxPool2d(2, 2) # (16, 15, 15)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3) # (32, 13, 13)
        self.pool2 = nn.MaxPool2d(2, 2) # (32, 6, 6)

        self.linear1 = nn.Linear(32 * 6 * 6, 128) # 128
        self.linear2 = nn.Linear(128, 10) # 10

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = out.view(-1, 32 * 6 * 6)
        out = F.relu(self.linear1(out))
        out = self.linear2(out)
        return out


def load_data(path:str):
    with open(path, 'r') as file:
        json_data = json.load(file)
        data_X = np.array([sample['Image'] for sample in json_data], dtype=np.uint8)
        data_Y = np.array([sample['Label'] for sample in json_data], dtype=np.int64)

    return data_X, data_Y

def plot_history(train_loss, train_acc, valid_loss, valid_acc):
    epochs = range(1, len(train_loss) + 1)

    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, valid_loss, label='Valid Loss')
    plt.title('Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Baseline/output/output_loss.png')
    plt.close()

    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, valid_acc, label='Valid Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Baseline/output/output_accuracy.png')
    plt.close()

def save_final_results(results, output_path='./Baseline/output/'):
    output_data = {
        "Learning rate": results['learning_rate'],  
        "Epoch": results['epochs'],
        "Batch size": results['batch_size'],
        "Final train accuracy": results['final_train_accuracy'],
        "Validation accuracy": results['final_val_accuracy'],
        "Final train loss": results['final_train_loss'],
        "Final validation loss": results['final_val_loss'],
    }
    with open(output_path + 'output_output.json', 'w') as f:  # 將訓練參數與結果寫入 JSON
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    full_data_X, full_data_Y = load_data('./train.json')

    # 切資料集
    total_size = full_data_X.shape[0]
    train_size = int(0.8 * total_size)
    indices = np.arange(total_size)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]

    # 定義資料增強與資料集
    train_transforms = transforms.Compose([
        transforms.ToTensor(),

        transforms.RandomHorizontalFlip(p=0.5), # 隨機水平翻轉 (50%機率)
        transforms.RandomApply([
            transforms.RandomRotation(15),  # 隨機旋轉 (-15度到15度)
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 隨機平移 (10%)
        ], p=0.5),  # 50%機率套用以上兩種變換
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # (mean, std) 正規化
    ])

    valid_transforms = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # (mean, std) 正規化
    ])

    train_dataset = MyDataset(full_data_X[train_indices], full_data_Y[train_indices], train_transforms)
    valid_dataset = MyDataset(full_data_X[valid_indices], full_data_Y[valid_indices], valid_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CNN().to('cuda:0')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    valid_acc_history = []

    for epoch in range(EPOCH):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for imgs, labels in train_loader:
            imgs = imgs.to('cuda:0')
            labels = labels.to('cuda:0')

            logits = model(imgs)

            preds = torch.argmax(logits, dim=-1)
            train_correct += (labels == preds).sum().item()

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        with torch.no_grad():
            for imgs, labels in valid_loader:
                imgs = imgs.to('cuda:0')
                labels = labels.to('cuda:0')

                logits = model(imgs)

                preds = torch.argmax(logits, dim=-1)
                valid_correct += (labels == preds).sum().item()

                loss = F.cross_entropy(logits, labels)
                valid_loss += loss.item()

        train_acc_history.append(train_correct / len(train_dataset))
        train_loss_history.append(train_loss / len(train_loader))
        valid_acc_history.append(valid_correct / len(valid_dataset))
        valid_loss_history.append(valid_loss / len(valid_loader))

        print("=======================================")
        print(f"Epoch {epoch+1}/{EPOCH}")
        print(f"Train acc: {train_acc_history[-1]:.4f})")
        print(f"Train loss: {train_loss_history[-1]:.4f})")
        print(f"Valid acc: {valid_acc_history[-1]:.4f})")
        print(f"Valid loss: {valid_loss_history[-1]:.4f})")
        print("=======================================")

        plot_history(train_loss_history, train_acc_history, valid_loss_history, valid_acc_history)
    
    results = {
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCH,
        'batch_size': BATCH_SIZE,
        'final_train_accuracy': train_acc_history[-1],
        'final_val_accuracy': valid_acc_history[-1],
        'final_train_loss': train_loss_history[-1],
        'final_val_loss': valid_loss_history[-1],
    }
    save_final_results(results)