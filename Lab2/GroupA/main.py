import numpy as np
import matplotlib.pyplot as plt
import json

TRAIN_FILE_PATH = './Group_A_train.csv'
TEST_FILE_PATH = './Group_A_test.csv'
OUTPUT_DIR = './output/'
CLASSES = [0, 1, 8, 9]
NUM_EPOCHS = 300
BATCH_SIZE = 40

class SoftmaxRegression:
    def __init__(self, input_size, num_classes, learning_rate=0.01):
        self.W = np.zeros((input_size, num_classes))
        self.b = np.zeros((1, num_classes))
        self.learning_rate = learning_rate
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.reg_lambda = 1e-4

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, x):
        y_hat = np.dot(x, self.W) + self.b
        probs = self.softmax(y_hat)
        return probs

    def compute_loss(self, y, y_hat): # cross-entropy loss
        m = y.shape[0]
        y_hat = np.clip(y_hat, 1e-12, 1.0)  # 防止 log(0)
        log_likelihood = -np.log(y_hat[range(m), np.argmax(y, axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss

    def compute_accuracy(self, y, y_hat):
        predictions = np.argmax(y_hat, axis=1)
        labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == labels)
        return accuracy
    
    def gradient_descent_update(self, x, y, y_hat):
        m = y.shape[0]
        delta = y_hat - y
        grad_W = np.dot(x.T, delta) / m
        grad_W += (self.reg_lambda) * self.W
        grad_b = np.sum(delta, axis=0, keepdims=True)
        return grad_W, grad_b
    
    def update_parameters(self, grad_W, grad_b):
        self.W -= self.learning_rate * grad_W
        self.b -= self.learning_rate * grad_b

    def shuffle_data(self, x, y):
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x_shuffled = np.array([x[i] for i in indices])
        y_shuffled = np.array([y[i] for i in indices])
        return x_shuffled, y_shuffled

    def train(self, x_train, y_train, x_val, y_val, num_epochs, batch_size):
        for epoch in range(num_epochs):
            epoch_train_loss = 0
            epoch_train_acc = 0
            num_batches = 0
            x_train, y_train = self.shuffle_data(x_train, y_train)
            for i in range(0, x_train.shape[0], batch_size):
                x_batch = x_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                y_hat_batch = self.forward(x_batch)
                loss = self.compute_loss(y_batch, y_hat_batch)
                accuracy = self.compute_accuracy(y_batch, y_hat_batch)
                grad_W, grad_b = self.gradient_descent_update(x_batch, y_batch, y_hat_batch)
                self.update_parameters(grad_W, grad_b)

                epoch_train_loss += loss
                epoch_train_acc += accuracy
                num_batches += 1

            avg_train_loss = epoch_train_loss / num_batches
            avg_train_acc = epoch_train_acc / num_batches
            y_val_hat = self.forward(x_val)
            val_loss = self.compute_loss(y_val, y_val_hat)
            val_accuracy = self.compute_accuracy(y_val, y_val_hat)

            self.train_loss_history.append(avg_train_loss)
            self.train_acc_history.append(avg_train_acc)
            self.val_loss_history.append(val_loss)
            self.val_acc_history.append(val_accuracy)

            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}')

        print('Final Results:')
        print(f'Train Loss={self.train_loss_history[-1]:.4f}, Train Acc={self.train_acc_history[-1]:.4f}, Val Loss={self.val_loss_history[-1]:.4f}, Val Acc={self.val_acc_history[-1]:.4f}')

    def predict(self, x):
        y_hat = self.forward(x)
        predictions = np.argmax(y_hat, axis=1)
        return predictions

def load_and_process_data(train_file, test_file):
    train_data = np.loadtxt(train_file, delimiter=',', skiprows=1)
    x_train = train_data[:, 1:]  # 從第 1 列開始取特徵（像素值）
    y_train = train_data[:, 0].astype(int)  # 第 0 列是標籤

    x_test = np.loadtxt(test_file, delimiter=',', skiprows=1)
    
    # 正規化到 [0, 1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test

def label_to_one_hot(labels, num_classes):
    one_hot = np.zeros((labels.shape[0], num_classes))
    for idx, label in enumerate(labels):
        one_hot[idx, CLASSES.index(label)] = 1
    return one_hot

def split_data(x, y, train_ratio=0.8):
    num_samples = x.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    train_size = int(num_samples * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    x_train = np.array([x[i] for i in train_indices])
    y_train = np.array([y[i] for i in train_indices])
    x_val = np.array([x[i] for i in val_indices])
    y_val = np.array([y[i] for i in val_indices])

    return x_train, y_train, x_val, y_val

def plot_draw(train_loss_history, val_loss_history, train_acc_history, val_acc_history):
    epochs = range(len(train_loss_history))

    plt.plot(epochs, train_loss_history, label='Train loss', color='blue')
    plt.plot(epochs, val_loss_history, label='Validation loss', color='orange')
    plt.title('GroupA_Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(OUTPUT_DIR + 'output_loss.png')
    plt.close()

    plt.plot(epochs, train_acc_history, label='Train Accuracy', color='blue')
    plt.plot(epochs, val_acc_history, label='Validation Accuracy', color='orange')
    plt.title('GroupA_Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(OUTPUT_DIR + 'output_accuracy.png')
    plt.close()

def save_final_results(results, predictions, filename='output.json'):
    output_data = {
        "Learning rate": results['learning_rate'],
        "Epochs": results['epochs'],
        "Batch size": results['batch_size'],
        "Final train accuracy": results['final_train_accuracy'],
        "Final val accuracy": results['final_val_accuracy'],
        "Final train loss": results['final_train_loss'],
        "Final val loss": results['final_val_loss'],
    }
    with open(OUTPUT_DIR + filename, 'w') as f:
        json.dump(output_data, f, indent=4)

    prediction_data = {
        "Predictions": predictions
    }
    with open(OUTPUT_DIR + 'test_set_prediction.json', 'w') as f:
        json.dump(prediction_data, f, indent=4)

if __name__ == '__main__':
    np.random.seed(123)
    x_train, y_train, x_test = load_and_process_data(TRAIN_FILE_PATH, TEST_FILE_PATH)
    y_train_one_hot = label_to_one_hot(y_train, len(CLASSES))
    x_train_split, y_train_split, x_val_split, y_val_split = split_data(x_train, y_train_one_hot, train_ratio=0.8)

    input_size = x_train.shape[1]

    model = SoftmaxRegression(input_size, len(CLASSES), 0.01)
    model.train(x_train_split, y_train_split, x_val_split, y_val_split, NUM_EPOCHS, BATCH_SIZE)

    plot_draw(model.train_loss_history, model.val_loss_history, model.train_acc_history, model.val_acc_history)

    final_train_loss = model.train_loss_history[-1]
    final_train_acc = model.train_acc_history[-1]
    final_val_loss = model.val_loss_history[-1]
    final_val_acc = model.val_acc_history[-1]

    predictions = model.predict(x_test).tolist()
    predictions = [CLASSES[p] for p in predictions]  # 映射回原始標籤

    save_final_results({
        'learning_rate': model.learning_rate,
        'epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'final_train_accuracy': final_train_acc,
        'final_val_accuracy': final_val_acc,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
    }, predictions)