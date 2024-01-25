import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from transformers import AutoModel, ResNetModel


class Add(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.visual_encoder = ResNetModel.from_pretrained('resnet-50')
        self.image_hidden_size = 2048
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(49152, 768)

        self.classifier = nn.Linear(max(self.text_encoder.config.hidden_size, self.image_hidden_size), num_labels)
        self.text_classifier = nn.Linear(self.text_encoder.config.hidden_size, num_labels)
        self.image_classifier = nn.Linear(self.image_hidden_size, num_labels)

    def forward(self, text, image):
        if (text is not None) and (image is not None):
            text_output = self.text_encoder(**text)
            text_feature = text_output.last_hidden_state[:, 0, :]
            # print('text_feature.shape', text_feature.shape)
            # print('text_feature.type', type(text_feature))

            img_feature = self.visual_encoder(**image).last_hidden_state.view(-1, 49, 2048).max(1)[0]
            # print('img_feature.shape', img_feature.shape)
            # print('img_feature.type', type(img_feature))

            padding = img_feature.size(1) - text_feature.size(1)
            text_feature = nn.functional.pad(text_feature, (0, padding), 'constant', 0)
            # print('text_feature.shape', text_feature.shape)
            # print('text_feature.type', type(text_feature))

            features = text_feature + img_feature
            # print('features.shape', features.shape)
            logits = self.classifier(features)
            # print('logits.shape', logits.shape)

            return logits

        elif text is not None:
            text_output = self.text_encoder(**text)
            # print(self.text_encoder.config.hidden_size)
            text_feature = text_output.last_hidden_state
            # print('text_feature.shape', text_feature.shape)

            logits = self.text_classifier(self.linear(self.flatten(text_feature)))
            return logits

        else:
            img_feature = self.visual_encoder(**image).last_hidden_state.view(-1, 49, 2048).max(1)[0]
            logits = self.image_classifier(img_feature)

            return logits


class Concat(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.visual_encoder = ResNetModel.from_pretrained('resnet-50')
        self.image_hidden_size = 2048
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(49152, 768)

        self.classifier = nn.Linear(self.text_encoder.config.hidden_size + self.image_hidden_size, num_labels)
        self.text_classifier = nn.Linear(self.text_encoder.config.hidden_size, num_labels)
        self.image_classifier = nn.Linear(self.image_hidden_size, num_labels)

    def forward(self, text, image):
        if (text is not None) and (image is not None):
            text_output = self.text_encoder(**text)
            text_feature = text_output.last_hidden_state[:, 0, :]
            # print('text_feature.shape', text_feature.shape)
            img_feature = self.visual_encoder(**image).last_hidden_state.view(-1, 49, 2048).max(1)[0]
            # print('img_feature.shape', img_feature.shape)
            features = torch.cat((text_feature, img_feature), 1)
            # print('features.shape', features.shape)

            logits = self.classifier(features)
            # print('logits.shape', logits.shape)

            return logits

        elif text is not None:
            text_output = self.text_encoder(**text)
            # print(self.text_encoder.config.hidden_size)
            text_feature = text_output.last_hidden_state
            # print('text_feature.shape', text_feature.shape)

            logits = self.text_classifier(self.linear(self.flatten(text_feature)))
            return logits

        else:
            img_feature = self.visual_encoder(**image).last_hidden_state.view(-1, 49, 2048).max(1)[0]
            logits = self.image_classifier(img_feature)

            return logits


class Attention(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.visual_encoder = ResNetModel.from_pretrained('resnet-50')
        self.image_hidden_size = 2048
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(49152, 768)
        self.attention = nn.MultiheadAttention(embed_dim=2048 + 768, num_heads=8)

        self.classifier = nn.Linear(self.text_encoder.config.hidden_size + self.image_hidden_size, num_labels)
        self.text_classifier = nn.Linear(self.text_encoder.config.hidden_size, num_labels)
        self.image_classifier = nn.Linear(self.image_hidden_size, num_labels)

    def forward(self, text, image):
        if (text is not None) and (image is not None):
            text_output = self.text_encoder(**text)
            text_feature = text_output.last_hidden_state[:, 0, :]
            # print('text_feature.shape', text_feature.shape)
            # [batchsize,2048,7,7] -> [batchsize,49,2048]
            img_feature = self.visual_encoder(**image).last_hidden_state.view(-1, 49, 2048).max(1)[0]
            # print('img_feature.shape', img_feature.shape)
            features = torch.cat((text_feature, img_feature), 1).unsqueeze(0)
            # print('features.shape', features.shape)

            # attention
            attention_output, _ = self.attention(Q=features, K=features, V=features)
            attention_output = attention_output.squeeze(0)  # 删除多余维度

            logits = self.classifier(attention_output)
            # print('logits.shape', logits.shape)

            return logits

        elif text is not None:
            text_output = self.text_encoder(**text)
            # print(self.text_encoder.config.hidden_size)
            text_feature = text_output.last_hidden_state
            # print('text_feature.shape', text_feature.shape)

            logits = self.text_classifier(self.linear(self.flatten(text_feature)))
            return logits

        else:
            img_feature = self.visual_encoder(**image).last_hidden_state.view(-1, 49, 2048).max(1)[0]
            logits = self.image_classifier(img_feature)

            return logits


