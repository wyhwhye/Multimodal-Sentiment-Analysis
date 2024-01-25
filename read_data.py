import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoFeatureExtractor
from transformers import AutoTokenizer


# 读取图片
def get_image(image_path):
    # image_path = './data/1.jpg'
    image = Image.open(image_path)
    return image


# 读取文本
def get_text(text_path):
    # text_path = './data/3.txt'
    with open(text_path, 'r', encoding='gb18030', errors='replace') as f:
        text = f.readline().strip()
        return text


# 获取dataset
def get_datasets(train_data, test_data):
    # 标签与id的映射
    label_to_id = {'negative': 0, 'neutral': 1, 'positive': 2}
    train_datasets = []
    test_datasets = []

    for guid, tag in train_data.values:
        # print(guid, tag)
        d = {
            'guid': int(guid),
            'tag': label_to_id[tag],
            'image': get_image(f'./data/{int(guid)}.jpg'),
            'text': get_text(f'./data/{int(guid)}.txt')
        }
        train_datasets.append(d)

    for guid, tag in test_data.values:
        # print(guid, tag)
        d = {
            'guid': int(guid),
            'tag': None,
            'image': get_image(f'./data/{int(guid)}.jpg'),
            'text': get_text(f'./data/{int(guid)}.txt')
        }
        test_datasets.append(d)

    return train_datasets, test_datasets


def collate_fn(datasets):
    feature_extractor = AutoFeatureExtractor.from_pretrained("resnet-50")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    guid = [data['guid'] for data in datasets]
    tag = [data['tag'] for data in datasets]
    if tag[0] is None:
        tag = None
    else:
        tag = torch.LongTensor(tag)
    image = [data['image'] for data in datasets]
    image = feature_extractor(image, return_tensors="pt")
    text = [data['text'] for data in datasets]
    text = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=64)

    return guid, tag, image, text


# 获取dataloader
def get_dataloader(train_datasets, test_datasets, batch_size):
    train_datasets, valid_datasets = train_test_split(train_datasets, test_size=0.1, random_state=42)

    train_dataloader = DataLoader(
        dataset=train_datasets,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    valid_dataloader = DataLoader(
        dataset=valid_datasets,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    test_dataloader = DataLoader(
        dataset=test_datasets,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_dataloader, valid_dataloader, test_dataloader


def read_data(batch_size):
    train_data = pd.read_csv('train.txt')
    test_data = pd.read_csv('test_without_label.txt')

    train_datasets, test_datasets = get_datasets(train_data, test_data)

    return get_dataloader(train_datasets, test_datasets, batch_size)
