import torch
from tqdm import tqdm, trange
from read_data import read_data


def predict(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    id_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    model = model.to(device)

    res = []

    for batch in tqdm(test_dataloader):
        model.eval()
        a, b_labels, b_imgs, b_text = batch
        b_text = b_text.to(device)
        b_imgs = b_imgs.to(device)

        with torch.no_grad():
            b_logits = model(text=b_text, image=b_imgs)
            b_logits = b_logits.detach().cpu()

        res += torch.argmax(b_logits, dim=-1).tolist()

    labels = [id_to_label[p] for p in res]

    return labels


if __name__ == '__main__':
    a, b, test_dataloader = read_data(32)
    file_path = 'test_without_label.txt'
    model = torch.load('./models/Attention.pth')

    replacements = predict(model, test_dataloader)

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 替换
    for i, line in enumerate(lines[1:]):
        if 'null' in line and i < len(replacements):
            lines[i + 1] = line.replace('null', replacements[i])

    # 写入
    new_file_path = 'test_with_label.txt'
    with open(new_file_path, 'w') as new_file:
        new_file.write(''.join(lines))

