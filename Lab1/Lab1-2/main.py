import numpy as np
import matplotlib.pyplot as plt
import json
import os

# read json
def readJson(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

# loss calculate
def mseLoss(x, y, w, b):
    y_pred = np.dot(x, w) + b
    return np.mean((y - y_pred.squeeze()) ** 2) # mse format

def drawRegression(x, y):
    # dot
    plt.scatter(x, y)
    # line
    plt.plot(x, x * w + b, label='regression line', color='red')
    plt.title('Regression')
    plt.legend(loc='best')
    # check dir is exist
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    # save chart
    plt.savefig(os.path.join(output_dir, 'lab1.2_result.png'))
    plt.show()
    plt.close()

def drawLoss(lossHistory):
    plt.plot(lossHistory, label='mse loss')
    plt.title('Loss')
    plt.legend(loc='best')
    # check dir is exist
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    # save chart
    plt.savefig(os.path.join(output_dir, 'lab1.2_loss.png'))
    plt.show()
    plt.close()

def writeJson(w, b, lossHistory, num_epochs, lr):
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # json
    data = {
        'final_weight': round(float(w), 2),
        'final_bias': round(float(b), 2),
        'loss_history': [round(float(x), 2) for x in lossHistory],
        'num_epochs': int(num_epochs),
        'learning_rate': round(float(lr), 2)
    }

    # write file
    json_path = os.path.join(output_dir, 'lab1.2_output.json')
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    np.random.seed(123)
    # read data
    data = readJson('lab1.2_train_data.json')
    x = np.array(data['x'])
    y = np.array(data['y'])

    # init
    w = np.random.uniform(-1, 1)
    b = np.random.uniform(-1, 1)
    lr = 0.001
    lossHistory = []
    maxEpoch = 50

    # calculate the first loss
    loss = mseLoss(x, y, w, b)
    lossHistory.append(loss)

    # train
    for epoch in range(maxEpoch):
        for i in range(len(x)):
            # predict
            y_pred = np.dot(x[i], w) + b

            # gradient descent
            w += lr * (y[i] - y_pred) * x[i]
            b += lr * (y[i] - y_pred)

        # loss update
        loss = mseLoss(x, y, w, b)
        lossHistory.append(loss)

    drawRegression(x, y)
    drawLoss(lossHistory)
    writeJson(w, b, lossHistory, maxEpoch, lr)