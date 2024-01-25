import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt


def draw(train_loss, train_acc, val_acc):
    plt.title('Training loss')
    plt.plot(train_loss, 'o')
    plt.xlabel('Iteration')
    plt.savefig('loss')
    plt.show()

    plt.title('Accuracy')
    plt.plot(train_acc, '-o', label='train_acc')
    plt.plot(val_acc, '-o', label='val_acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.savefig('acc')
    plt.show()


def train(
        train_dataloader,
        valid_dataloader,
        model,
        epochs,
        weight_decay,
        learning_rate,
        text_only,
        image_only
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()

    model.train()

    train_loss = []  # 每个epoch的训练平均损失
    val_loss = []  # 每个epoch的验证平均损失
    train_acc = []  # 训练集accuracy
    train_f1 = []  # 训练集f1
    train_precision = []  # 训练集precision
    train_recall = []  # 训练集recall
    val_acc = []  # 验证集accuracy
    val_f1 = []  # 验证集f1
    val_precision = []  # 验证集precision
    val_recall = []  # 验证集recall

    for epoch in range(epochs):
        train_pred = []  # 训练集预测标签
        train_label = []  # 训练集真实标签

        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Batch'):
            # print('batch', batch)
            a, b_labels, b_imgs, b_text = batch
            b_labels = b_labels.to(device)
            b_imgs = b_imgs.to(device)
            b_text = b_text.to(device)

            model.zero_grad()

            if text_only:
                b_logits = model(text=b_text, image=None)
            elif image_only:
                b_logits = model(text=None, image=b_imgs)
            else:
                b_logits = model(text=b_text, image=b_imgs)
            # print('b_logits', b_logits.shape)

            loss = criterion(b_logits, b_labels)
            loss.backward()

            optimizer.step()

            train_loss.append(loss.item())
            b_logits = torch.max(b_logits, 1)[1]
            train_pred += b_logits.detach().cpu().tolist()
            train_label += b_labels.detach().cpu().tolist()

        acc = accuracy_score(train_pred, train_label)
        f1 = f1_score(train_pred, train_label, average='weighted')
        recall = recall_score(train_pred, train_label, average='weighted')
        precision = precision_score(train_pred, train_label, average='weighted')

        train_acc.append(acc)
        train_f1.append(f1)
        train_precision.append(precision)
        train_recall.append(recall)

        print('epoch =', epoch + 1)
        print("train_loss = {}\ntrain_acc = {}\t train_f1 = {}\ntrain_recall={}\t train_precision={}"
              .format(train_loss[-1], acc, f1, recall, precision))

        with torch.no_grad():
            model.eval()
            val_pred = []  # 验证集预测标签
            val_target = []  # 验证集真实标签

            for batch in tqdm(valid_dataloader):
                a, b_labels, b_imgs, b_text = batch
                b_labels = b_labels.to(device)
                b_imgs = b_imgs.to(device)
                b_text = b_text.to(device)

                if text_only:
                    b_logits = model(text=b_text, image=None)
                elif image_only:
                    b_logits = model(text=None, image=b_imgs)
                else:
                    b_logits = model(text=b_text, image=b_imgs)

                v_loss = criterion(b_logits, b_labels)

                val_loss.append(v_loss.item())
                b_logits = torch.max(b_logits, 1)[1]

                val_pred += b_logits.detach().cpu().tolist()
                val_target += b_labels.detach().cpu().tolist()

            acc = accuracy_score(val_pred, val_target)
            f1 = f1_score(val_pred, val_target, average='weighted')
            recall = recall_score(val_pred, val_target, average='weighted')
            precision = precision_score(val_pred, val_target, average='weighted')

            val_acc.append(acc)
            val_f1.append(f1)
            val_precision.append(precision)
            val_recall.append(recall)

            print("val_loss = {}\nval_acc = {}\t val_f1 = {}\nval_recall={}\t val_precision={}"
                  .format(val_loss[-1], acc, f1, recall, precision))

    print('\n***Training completed***\n')
    # draw(train_loss, train_acc, val_acc)
