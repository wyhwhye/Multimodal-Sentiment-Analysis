from read_data import read_data
from model import Concat, Add, Attention
from train import train
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser(description="Seq2Seq Training Script")

parser.add_argument("--model", type=str, default='Concat', help="Choose a model among Concat, Add, Attention")
# parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation")
parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer")
parser.add_argument('--text_only', type=bool, default=False, help='only use text?')
parser.add_argument('--image_only', type=bool, default=False, help='only use image?')
parser.add_argument('--do_predict', type=bool, default=False, help='do predict?')

args = parser.parse_args()


# 随机种子
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

# 数据
train_dataloader, valid_dataloader, test_dataloader = read_data(batch_size=32)

# 模型
if args.model == "Concat":
    model = Concat()
elif args.model == "Add":
    model = Add()
elif args.model == "Attention":
    model = Attention()

# 训练
train(
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    model=model,
    epochs=args.epochs,
    weight_decay=args.weight_decay,
    learning_rate=args.learning_rate,
    text_only=args.text_only,
    image_only=args.image_only
)
torch.save(model, './models/myModel.pth')
