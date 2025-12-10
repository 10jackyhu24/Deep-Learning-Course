問題1: 一開始w初始化使用self.W[i] = np.zeros((layer_sizes[i], layer_sizes[i+1]))，發現模型訓練很快就早停了。
解法: 改成self.W[i] = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01

問題2: 在初始化權重時，使用了錯誤的維度 np.zeros((input_size, layer_sizes[i+1]))，導致所有層的權重矩陣第一維都是 input_size。
解法: 改成 np.zeros((layer_sizes[i], layer_sizes[i+1]))，讓每層的權重矩陣根據前一層的輸出大小來決定。