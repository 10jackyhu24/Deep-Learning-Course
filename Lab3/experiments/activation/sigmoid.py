import numpy as np
import matplotlib.pyplot as plt
import json

TRAIN_FILE_PATH = './Group_B_train.csv'
TEST_FILE_PATH = './Group_B_test.csv'
OUTPUT_DIR = './output/'
CLASSES = [0, 2, 4, 6]
NUM_EPOCHS = 1000
BATCH_SIZE = 40

class SoftmaxRegression:
    def __init__(self, input_size, num_classes, learning_rate=0.01, patience=10, hidden_layers=[]):
        self.hidden_layers = hidden_layers  # 隱藏層
        self.learning_rate = learning_rate  # 學習率
        self.train_loss_history = []  # training loss 歷史紀錄
        self.train_acc_history = []  # training accuracy 歷史紀錄
        self.val_loss_history = []  # validation loss 歷史紀錄
        self.val_acc_history = []  # validation accuracy 歷史紀錄
        self.reg_lambda = 1e-4  # 正則化強度
        self.patience = patience  # 早停容忍度
        self.end_epoch = 0  # 記錄最終 epoch 數
        self.end_batch = 0  # 記錄最終 batch 數

        # 初始化權重和偏差
        layer_sizes = [input_size] + hidden_layers + [num_classes]
        num_layers = len(layer_sizes) - 1
        
        # 創建空陣列
        self.W = np.empty(num_layers, dtype=object)
        self.b = np.empty(num_layers, dtype=object)
        
        for i in range(num_layers):
            self.W[i] = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            self.b[i] = np.zeros((1, layer_sizes[i+1]))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # 數值穩定的 exp，對每列減去最大值
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)  # 每列除以總和，得到機率分布
    
    def forward(self, x):
        self.activations = [x]  # 儲存每層的激活值
        self.z_values = []  # 儲存每層的輸出
        
        current_input = x
        # 前向傳播所有隱藏層
        for i in range(len(self.hidden_layers)):
            z = np.dot(current_input, self.W[i]) + self.b[i]
            self.z_values.append(z)
            a = self.sigmoid(z)
            self.activations.append(a)
            current_input = a
        
        # 輸出層
        z = np.dot(current_input, self.W[-1]) + self.b[-1]
        self.z_values.append(z)
        probs = self.softmax(z)
        self.activations.append(probs)
        
        return probs

    def compute_loss(self, y, y_hat): # cross-entropy loss
        m = y.shape[0]  # 批次大小
        y_hat = np.clip(y_hat, 1e-12, 1.0)  # 防止 log(0) 導致 -inf
        log_likelihood = -np.log(y_hat[range(m), np.argmax(y, axis=1)])  # 取出每筆樣本真實類別的機率並取 -log
        loss = np.sum(log_likelihood) / m  # 平均交叉熵
        return loss

    def compute_accuracy(self, y, y_hat):
        predictions = np.argmax(y_hat, axis=1)  # 從機率取最大的 index
        labels = np.argmax(y, axis=1)  # 真實類別的 index
        accuracy = np.mean(predictions == labels)  # 比較並計算平均(正確率)
        return accuracy

    def gradient_descent_update(self, x, y, y_hat):
        m = y.shape[0]  # 批次大小

        # 反向傳播
        grads_W = []
        grads_b = []
        
        delta = y_hat - y  # softmax + cross-entropy 的誤差項
       
        for i in range(len(self.W) - 1, -1, -1):
            grad_W = np.dot(self.activations[i].T, delta) / m  # 計算 W 的梯度，並做平均
            grad_W += self.reg_lambda * self.W[i]  # 加上 L2 正則化梯度
            grad_b = np.sum(delta, axis=0, keepdims=True) / m  # 計算 b 的梯度
        
            grads_W.insert(0, grad_W)
            grads_b.insert(0, grad_b)
            
            # 如果不是第一層，計算前一層的 delta
            if i > 0:
                delta = np.dot(delta, self.W[i].T) * self.sigmoid_derivative(self.z_values[i-1])
        return grads_W, grads_b
    
    def update_parameters(self, grad_W, grad_b):
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - self.learning_rate * grad_W[i]  # W = W - lr * grad_W
            self.b[i] = self.b[i] - self.learning_rate * grad_b[i]  # b = b - lr * grad_b

    def shuffle_data(self, x, y):
        indices = np.arange(x.shape[0])  # index 從 0 到 N-1
        np.random.shuffle(indices)  # 隨機打亂 index
        x_shuffled = np.array([x[i] for i in indices])  # 根據 index 重排 x
        y_shuffled = np.array([y[i] for i in indices])  # 根據 index 重排 y
        return x_shuffled, y_shuffled

    def train(self, x_train, y_train, x_val, y_val, num_epochs, batch_size):
        min_val_loss = float('inf')  # 初始化最小驗證損失
        patience_counter = 0  # 早停計數器
        for epoch in range(num_epochs):
            epoch_train_loss = 0  # 當前的 epoch 的train loss
            epoch_train_acc = 0  # 當前的 epoch 的train accuracy
            num_batches = 0  # 2; 當前的 epoch 的 batch
            x_train, y_train = self.shuffle_data(x_train, y_train)  # 打亂資料
            for i in range(0, x_train.shape[0], batch_size):  # 以 batch_size 步進切分資料
                x_batch = x_train[i:i+batch_size]  # 取得當前 batch 的 x
                y_batch = y_train[i:i+batch_size]  # 取得當前 batch 的 y

                y_hat_batch = self.forward(x_batch)  # 前向計算取得機率
                loss = self.compute_loss(y_batch, y_hat_batch)  # 計算 loss
                accuracy = self.compute_accuracy(y_batch, y_hat_batch)  # 計算準確率
                grad_W, grad_b = self.gradient_descent_update(x_batch, y_batch, y_hat_batch)  # 計算梯度
                self.update_parameters(grad_W, grad_b)  # 更新參數

                epoch_train_loss += loss  # 累加訓練損失
                epoch_train_acc += accuracy  # 累加訓練準確率
                num_batches += 1

            avg_train_loss = epoch_train_loss / num_batches  # 平均訓練損失
            avg_train_acc = epoch_train_acc / num_batches  # 平均訓練準確率
            y_val_hat = self.forward(x_val)  # 在整個驗證集上做前向推論
            val_loss = self.compute_loss(y_val, y_val_hat)  # 計算驗證 loss
            val_accuracy = self.compute_accuracy(y_val, y_val_hat)  # 計算驗證集 accuracy

            self.train_loss_history.append(avg_train_loss)  # 儲存 train loss
            self.train_acc_history.append(avg_train_acc)  # 儲存 train accuracy
            self.val_loss_history.append(val_loss)  # 儲存 val loss
            self.val_acc_history.append(val_accuracy)  # 儲存 val accuracy

            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}')

            # 早停機制
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                patience_counter = 0  # 重置計數器
            else:
                patience_counter += 1

            self.end_epoch = epoch  # 記錄最終 epoch 數
            self.end_batch = num_batches  # 記錄最終 batch 數
            if patience_counter >= self.patience:
                print(f'Early stopping at epoch {epoch}')
                break

        print('Final Results:')
        print(f'Train Loss={self.train_loss_history[-1]:.4f}, Train Acc={self.train_acc_history[-1]:.4f}, Val Loss={self.val_loss_history[-1]:.4f}, Val Acc={self.val_acc_history[-1]:.4f}')

    def predict(self, x):
        y_hat = self.forward(x)  # 前向得到機率
        predictions = np.argmax(y_hat, axis=1)  # 取最大值 index 作為預測類別
        return predictions  # 回傳預測整數 index

def load_and_process_data(train_file, test_file):
    train_data = np.loadtxt(train_file, delimiter=',', skiprows=1)  # 以 numpy 載入 CSV，跳過 第 1 個row
    x_train = train_data[:, 1:]  # 取得 x，跳過第 1 個 column
    y_train = train_data[:, 0].astype(int)  # 取得標籤(第 1 個 column)，轉為整數型別

    x_test = np.loadtxt(test_file, delimiter=',', skiprows=1)  # 載入測試資料
    
    # 正規化到 [0, 1]
    x_train = x_train / 255.0  # 將像素值縮放到 0-1
    x_test = x_test / 255.0  # 測試資料同樣正規化

    return x_train, y_train, x_test

def label_to_one_hot(labels, num_classes):
    one_hot = np.zeros((labels.shape[0], num_classes))
    for idx, label in enumerate(labels):
        one_hot[idx, CLASSES.index(label)] = 1  # 將原始標籤映射到 CLASSES 的index並設為 1
    return one_hot

def split_data(x, y, train_ratio=0.8):
    num_samples = x.shape[0]  # 樣本總數
    indices = np.arange(num_samples)  # 建立 index 陣列
    np.random.shuffle(indices)  # 打亂 index

    train_size = int(num_samples * train_ratio)  # 計算訓練集大小
    train_indices = indices[:train_size]  # 取出訓練 index
    val_indices = indices[train_size:]  # 其餘為驗證 index

    # 從index取得資料
    x_train = np.array([x[i] for i in train_indices])
    y_train = np.array([y[i] for i in train_indices])
    x_val = np.array([x[i] for i in val_indices])
    y_val = np.array([y[i] for i in val_indices])

    return x_train, y_train, x_val, y_val

def plot_draw(train_loss_history, val_loss_history, train_acc_history, val_acc_history):
    epochs = range(len(train_loss_history)) # 根據歷史紀錄長度建立 epoch 範圍

    plt.plot(epochs, train_loss_history, label='Train loss', color='blue')  # 繪製
    plt.plot(epochs, val_loss_history, label='Validation loss', color='orange')  # 繪製
    plt.title('GroupB_Loss')  # 圖片標題
    plt.xlabel('Epochs')  # x 標籤
    plt.ylabel('Loss')  # y 標籤
    plt.legend()  # 顯示圖例
    plt.savefig(OUTPUT_DIR + 'output_loss.png')  # 儲存圖檔
    plt.close()  # 關閉圖表

    plt.plot(epochs, train_acc_history, label='Train Accuracy', color='blue')  # 繪製
    plt.plot(epochs, val_acc_history, label='Validation Accuracy', color='orange')  # 繪製
    plt.title('GroupB_Accuracy')  # 圖片標題
    plt.xlabel('Epochs')  # x 標籤
    plt.ylabel('Accuracy')  # y 標籤
    plt.legend()  # 顯示圖例
    plt.savefig(OUTPUT_DIR + 'output_accuracy.png')  # 儲存準確率圖檔
    plt.close()  # 關閉圖表

def save_final_results(results, predictions, filename='output.json'):
    # 輸出資料
    output_data = {
        "Learning rate": results['learning_rate'],  
        "Epoch": results['epochs'],
        "Batch size": results['batch_size'],
        "Final train accuracy": results['final_train_accuracy'],
        "Validation accuracy": results['final_val_accuracy'],
        "Final train loss": results['final_train_loss'],
        "Final validation loss": results['final_val_loss'],
    }
    with open(OUTPUT_DIR + filename, 'w') as f:  # 將訓練參數與結果寫入 JSON
        json.dump(output_data, f, indent=4)

    prediction_data = {
        "Predictions": predictions  # 預測結果
    }
    with open(OUTPUT_DIR + 'test_set_prediction.json', 'w') as f:  # 儲存測試集預測
        json.dump(prediction_data, f, indent=4)

if __name__ == '__main__':
    np.random.seed(123)  # 設定亂數種子以利結果重現
    x_train, y_train, x_test = load_and_process_data(TRAIN_FILE_PATH, TEST_FILE_PATH)  # 載入並做預處理
    y_train_one_hot = label_to_one_hot(y_train, len(CLASSES))  # 將原始標籤轉為 one-hot
    x_train_split, y_train_split, x_val_split, y_val_split = split_data(x_train, y_train_one_hot, train_ratio=0.8)  # 分割訓練/驗證集

    input_size = x_train.shape[1]

    model = SoftmaxRegression(input_size, len(CLASSES), 0.01, 10, hidden_layers=[128])  # 建立模型
    model.train(x_train_split, y_train_split, x_val_split, y_val_split, NUM_EPOCHS, BATCH_SIZE)  # 訓練

    plot_draw(model.train_loss_history, model.val_loss_history, model.train_acc_history, model.val_acc_history)  # 畫圖並儲存

    end_epoch = model.end_epoch
    end_batch = model.end_batch
    final_train_loss = model.train_loss_history[-1]
    final_train_acc = model.train_acc_history[-1]
    final_val_loss = model.val_loss_history[-1]
    final_val_acc = model.val_acc_history[-1]

    predictions = model.predict(x_test).tolist()  # 對測試集預測並轉為 list
    predictions = [CLASSES[p] for p in predictions]  # 將模型輸出的 index 轉回原始標籤

    save_final_results({
        'learning_rate': model.learning_rate,
        'epochs': end_epoch,
        'batch_size': end_batch,
        'final_train_accuracy': final_train_acc,
        'final_val_accuracy': final_val_acc,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
    }, predictions)