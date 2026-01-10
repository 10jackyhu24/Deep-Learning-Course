import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

LEARNING_RATE = 0.001
EPOCH = 150
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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        
        return out
    

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # input (3, 32, 32)
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # (32, 32, 32)
        self.bn1 = nn.BatchNorm2d(32)
        
        # First Residual Block
        self.res_block1 = ResidualBlock(32, 32, stride=1) # (32, 32, 32)
        self.res_block2 = ResidualBlock(32, 64, stride=2) # (64, 16, 16)
        
        # Second Residual Block
        self.res_block3 = ResidualBlock(64, 64, stride=1) # (64, 16, 16)
        self.res_block4 = ResidualBlock(64, 128, stride=2) # (128, 8, 8)
        
        # Third Residual Block
        self.res_block5 = ResidualBlock(128, 128, stride=1) # (128, 8, 8)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # (128, 1, 1)
        
        # Fully connected layers with dropout
        self.dropout1 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(128, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        # Initial convolution
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.res_block3(out)
        out = self.res_block4(out)
        out = self.res_block5(out)
        
        # Global average pooling
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        
        # Fully connected layers
        out = self.dropout1(out)
        out = F.relu(self.bn_fc(self.linear1(out)))
        out = self.dropout2(out)
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
    plt.savefig('Model B/output/output_loss.png')
    plt.close()

    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, valid_acc, label='Valid Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Model B/output/output_accuracy.png')
    plt.close()

def save_final_results(results, output_path='./Model B/output/'):
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
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 顏色抖動
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))  # 高斯模糊
        ], p=0.3),
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
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Learning rate scheduler - 每30個epoch減少學習率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

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
        
        # 更新學習率
        scheduler.step()

        print("=======================================")
        print(f"Epoch {epoch+1}/{EPOCH}")
        print(f"Train acc: {train_acc_history[-1]:.4f})")
        print(f"Train loss: {train_loss_history[-1]:.4f})")
        print(f"Valid acc: {valid_acc_history[-1]:.4f})")
        print(f"Valid loss: {valid_loss_history[-1]:.4f})")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
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