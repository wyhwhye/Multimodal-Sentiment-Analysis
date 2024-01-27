# Multimodal-Sentiment-Analysis
 当代人工智能大作业，多模态情感分析。


## 文件结构

```python
│  train.txt # 训练数据标签
│  test_without_label.txt # 测试集文件
│  requirements.txt # 依赖
│  README.md # README
│  main.py # 脚本文件
│  model.py # 模型
│  predict.py # 预测
│  read_data.py # 读入数据
│  train.py # 训练
│  预测结果.txt # 预测结果  
├─data # 图片和文本数据
├─bert-based-uncased # bert模型
├─resnet-50 # resnet模型
```


## SetUp

- torch==1.11
- numpy==1.21.6
- transformers==4.36.2
- matplotlib==3.5.3
- tqdm==4.66.1
- scikit-learn==1.3.2
- pandas==1.3.5
- pillow==9.1.1


执行以下代码安装依赖：

```shell
pip install -r requirements.txt
```


## 运行代码

1. **训练模型**

    使用下面的命令来运行脚本训练模型，参数可自行调整（所给为默认参数）。

    ```sh
    python main.py --model Attention --learning_rate 1e-5 --epochs 10 --weight_decay 0.01 --text_only False --image_only False
    ```

    参数说明：

    
    *   `--model`：模型名称，可选三个：Add，Concat，Attention。
    *   `--learning_rate`：学习率，用于优化器。
    *   `--epochs`：训练的总周期数。
    *   `--weight_decay`：权重衰减，用于优化器。
    *   `--text_only`：只使用文本做消融实验。
    *   `--image_only`：只使用图像做消融实验。


2. **预测结果**

    ```shell
    python predict.py
    ```


## 实验结果

|  模型    |  text_and_image    |   text_only    |   image_only    |
| ---- | ---- | ---- | ---- |
|   Add   |  73.67    | 73.08%     |  61.25%    |
|  Concat    |    73.75%  |  71.58%    |    58.5%  |
|  Attention    |   **73.8%**   |   72.08%   |   56.5%   |



## 参考

[1] Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." *arXiv preprint arXiv:1810.04805* (2018).

[2] He, Kaiming, et al. "Deep residual learning for image recognition." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

[3] Vaswani, Ashish, et al. "Attention is all you need." *Advances in neural information processing systems* 30 (2017).

[4] Quoc-Tuan Truong, Hady W. Lauw. VistaNet: Visual Aspect Attention Network for Multimodal Sentiment Analysis

[liyunfan1223/multimodal-sentiment-analysis: 该仓库存放了多模态情感分析实验的配套代码。 (github.com)](https://github.com/liyunfan1223/multimodal-sentiment-analysis)

[YeexiaoZheng/Multimodal-Sentiment-Analysis: 多模态情感分析——基于BERT+ResNet的多种融合方法 (github.com)](https://github.com/YeexiaoZheng/Multimodal-Sentiment-Analysis)





