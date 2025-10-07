import numpy as np
import matplotlib.pyplot as plt
import json
import os

def fibMatrix(n: int, mod: int) -> int:
    if not n: # if n == 0
        return 0 # a0 is 0
    
    # the formula in the problem, dtype=np.int64 (Reduce floating-point errors)
    fib = np.array([[1, 1], [1, 0]], dtype=np.int64)
     # the first matrix
    result = np.array([[1], [0]], dtype=np.int64)
    # calculate
    for _ in range(n - 1):
        result = fib @ result
    return result[0][0] % mod

def writeJson(mod: int, fib: list):
    # check dir is exist
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # json
    data = {
        "mod": mod,
        "pisano_period": fib
    }

    # white file to dir/lab1.1_output.json
    json_path = os.path.join(output_dir, 'lab1.1_output.json')
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def draw_and_save(fib: list):
    plt.plot(fib)
    plt.title('F(n) mod F(n-1)')
    plt.xlabel('n')
    plt.ylabel('F(n) mod F(n-1)')
    plt.grid(True)
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'lab1.1.png'))
    plt.show()
    plt.close()

if __name__ == '__main__':
    mod = 5
    fib = [fibMatrix(i, mod) for i in range(60)]
    print(fib)
    fib = [int(x) for x in fib]
    writeJson(mod, fib)
    draw_and_save(fib)