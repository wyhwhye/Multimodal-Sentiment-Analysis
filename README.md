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


执行以下代码安装依赖：

```shell
pip install -r requirments.txt
```

## 文件结构

由于不能上传超过25M的文件，所以这里的bert，bert_tokenizer, resnet-50， bertSelf.pt没法上传，但是融合模型会用到这些文件，所以所有的文件会通过邮件发送，发送的格式与此处的文件结构相同。虽然bert和resnet预训练模型可以通过代码直接下载，但我在实验中发现经常会因为网络原因下载失败，并且下载一个bert模型需要一定的时间，所以为了保险起见，选择通过邮件发送模型。最后要麻烦助教在检查的时候将这些数据上传到自己的设备或服务器中～

```python
|-- bert # 这个文件夹包含了BERT基础模型的的预训练参数
    |-- .DS_Store # 无需在意
    |-- bert_config.json  # BERT模型的参数
    |-- bert_model.ckpt.data-00000-of-00001  # all run shs
    |-- bert_model.ckpt.index
    |-- bert_model.ckpt.meta #
    |-- config.json # includes all model implementations
    |-- pytorch_model.bin # pytorch bert的模型
|-- bert_tokenizer # BERT模型的分词器相关代码
    |-- .DS_Store # 无需在意
    |-- special_tokens_map.json
    |-- tokenizer.json
    |-- tokenizer_config.json
    |-- vocab.txt 
|-- resnet-50 # 这个文件夹包含了ResNet-50基础模型的的预训练参数
|-- 实验五数据 # 本次实验用到的数据
    ｜-- data # 文本与图像
|-- .gitattributes # git lfs 用于上传大文件的一些代码
|-- README.md # 本项目的介绍
|-- bertSelf.pt #训练好的多模态模型，用于预测
|-- data_util.py # 数据处理的文件，里面包含了构建dataset类与dataloader的代码
|-- df_for_test.csv # 整理好的用于预测数据集
|-- df_for_train.csv # 整理好的用于训练的数据集
|-- main.py # 主函数
|-- model.py  # 构建多模态模型的代码
|-- prediction.py  # 用于预测的代码
|-- requirements.txt # 创建好云服务器后需要的依赖
|-- test_with_labels.txt # 本次实验的提交答案
|-- train.py # 用于训练的模型的代码
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







