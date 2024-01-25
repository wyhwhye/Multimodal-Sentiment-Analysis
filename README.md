# Multimodal-Sentiment-Analysis
 当代人工智能大作业，多模态情感分析。

## SetUp

- torch==1.13.1
- numpy==1.21.6
- transformers==4.30.2
- matplotlib==3.5.3
- tqdm==4.66.1
- scikit-learn==1.0.2
- pandas==1.3.5
- pillow==9.5.0


执行以下代码安装：

```shell
pip install -r requirments.txt
```

## 文件结构

```python
├─data # 图片和文本数据
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
├─bert-based-uncased # bert模型
├─resnet-50 # resnet模型
```

## 运行代码

1. **训练模型**

    使用下面的命令来运行脚本，参数可自行调整（所给为默认参数）。

    ```sh
    python main.py --learning_rate 0.001 --epochs 10 --weight_decay 0.01 --batch_size 64
    ```

    参数说明：

    *   `--learning_rate`：学习率，用于优化器。
    *   `--epochs`：训练的总周期数。
    *   `--weight_decay`：权重衰减，用于优化器。
    *   `--batch_size`：训练和验证时的批处理大小。


2. **预测结果**

    ```shell
    python predict.py
    ```

## 参考

[1]  Zhen Li, Bing Xu, Conghui Zhu, and Tiejun Zhao. CLMLF:a contrastive learning and multi-layer fusion method for multimodal sentiment detection. In Findings of the Association for Computational Linguistics: NAACL 2022. Association for Com- putational Linguistics, 2022.

[2] Kaiming He, X. Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016.

[3] Chuhan Wu, Fangzhao Wu, Tao Qi, Yongfeng Huang, and Xing Xie. Fastformer: Additive attention can be all you need. CoRR, abs/2108.09084, 2021.

[4] Quoc-Tuan Truong, Hady W. Lauw. VistaNet: Visual Aspect Attention Network for Multimodal Sentiment Analysis

[5] Guimin Hu , Ting-En Lin, Yi Zhao , Guangming Lu , Yuchuan Wu, Yongbin Li.UniMSE: Towards Unified Multimodal Sentiment Analysis and Emotion Recognition

https://github.com/Link-Li/CLMLF

https://github.com/liyunfan1223/multimodal-sentiment-analysis







