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

由于不能上传超过25M的文件，所以这里的bert，bert_tokenizer, resnet-50， bertSelf.pt没法上传，但是融合模型会用到这些文件，所以所有的文件会通过邮件发送，发送的格式与此处的文件结构相同。虽然bert和resnet预训练模型可以通过代码直接下载，但我在实验中发现经常会因为网络原因下载失败，并且下载一个bert模型需要一定的时间，所以为了保险起见，选择通过邮件发送模型。最后要麻烦助教在检查的时候将这些数据上传到自己的设备或服务器中～

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
├─resnet-50 # re's'm
```

## 训练模型

### 查看可设置的参数

```python
python ./main.py -h
```

```python
--model MODEL         #选择使用的模型
--lr LR               #设置学习率
--weight_decay WEIGHT_DECAY
                        #设置权重衰减
--epochs EPOCHS       #设置训练轮数
--batch_size BATCH_SIZE
                        #批量大小
--warmup WARMUP       #预热学习率步数
```

### 训练模型

首先进入文件夹Multimodal-Sentiment-Analysis

```shell
cd Multimodal-Sentiment-Analysis
```

#### 多模态融合模型

一共有4个模型分别是**Concatenation, Additive Attention, Multi-Layer Fusion, CL-Multi-Layer Fusion**,具体可以见实验报告。\
Multi-Layer Fusion

```shell
python ./main.py --model mlf
```

Concatenation

```shell
python ./main.py --model concat
```

Additive Attention

```shell
python ./main.py --model additive
```

CL-Multi-Layer Fusion

```shell
python ./main.py --model cl
```

#### 消融模型

消融是对于Multi-Layer Fusion而言的
text—only

```shell
python ./main.py --model text_only
```

image—only

```shell
python ./main.py --model image_only
```

### 预测

利用之前训练好的模型预测(bertSelf.pt)

```shell
python ./main.py --model test
```

## 参考

[1]  Zhen Li, Bing Xu, Conghui Zhu, and Tiejun Zhao. CLMLF:a contrastive learning and multi-layer fusion method for multimodal sentiment detection. In Findings of the Association for Computational Linguistics: NAACL 2022. Association for Com- putational Linguistics, 2022.

[2] Kaiming He, X. Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016.

[3] Chuhan Wu, Fangzhao Wu, Tao Qi, Yongfeng Huang, and Xing Xie. Fastformer: Additive attention can be all you need. CoRR, abs/2108.09084, 2021.

[4] Quoc-Tuan Truong, Hady W. Lauw. VistaNet: Visual Aspect Attention Network for Multimodal Sentiment Analysis

[5] Guimin Hu , Ting-En Lin, Yi Zhao , Guangming Lu , Yuchuan Wu, Yongbin Li.UniMSE: Towards Unified Multimodal Sentiment Analysis and Emotion Recognition

https://github.com/Link-Li/CLMLF

https://github.com/liyunfan1223/multimodal-sentiment-analysis







