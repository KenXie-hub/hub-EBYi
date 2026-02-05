import torch
import numpy as np
from transformers import BertTokenizer
import pandas as pd
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm

df = pd.read_csv('toutiao_cat_data.csv')
df = df.head(1600)  # 时间原因，我只取了1600条训练
np.random.seed(112)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                     [int(.8*len(df)), int(.9*len(df))])  # 拆分为训练集、验证集和测试集，比例为 80:10:10。

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

labels = {'news_story':0,
          'news_culture':1,
          'news_entertainment':2,
          'news_sports':3,
          'news_finance':4,
          'news_house':5,
          'news_car':6,
          'news_edu':7,
          'news_tech':8,
          'news_military':9,
          'news_travel':10,
          'news_world':11,
          'stock':12,
          'news_agriculture':13,
          'news_game':14
          }

# 反向映射：从数字ID转回标签文本
id_to_label = {v: k for k, v in labels.items()}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [labels[label] for label in df['label']]
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length = 512,
                                truncation=True,
                                return_tensors="pt")
                      for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese',num_labels=15)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 15)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

def train(model, train_data, val_data, learning_rate, epochs, batch_size):
    train, val = Dataset(train_data), Dataset(val_data)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0
        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                output = model(input_id, mask)
                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
              | Val Loss: {total_loss_val / len(val_data): .3f} 
              | Val Accuracy: {total_acc_val / len(val_data): .3f}''')

EPOCHS = 10
model = BertClassifier()
LR = 1e-6
Batch_Size = 16
train(model, df_train, df_val, LR, EPOCHS, Batch_Size)
torch.save(model.state_dict(), 'BERT-toutiao.pt')

def evaluate(model, test_data):
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=16)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')

evaluate(model, df_test)

def predict_single_text(model, tokenizer, text, device):
    """
    输入单条文本，返回模型预测的标签
    """
    model.eval()
    # 对输入文本进行编码
    encoding = tokenizer(
        text,
        padding='max_length',
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )
    input_id = encoding['input_ids'].squeeze(1).to(device)
    mask = encoding['attention_mask'].to(device)
    
    # 模型推理
    with torch.no_grad():
        output = model(input_id, mask)
        pred_label_id = output.argmax(dim=1).item()
    
    # 映射回标签文本
    pred_label = id_to_label[pred_label_id]
    return pred_label

# ------------------------------
# 测试示例
# ------------------------------
if __name__ == "__main__":
    # 加载模型
    model = BertClassifier()
    model.load_state_dict(torch.load('BERT-toutiao.pt'))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
    
    # 测试文本列表
    test_texts = [
        "京城最值得你来场文化之旅的博物馆保利集团,马未都,中国科学技术馆,博物馆,新中国",
        "CBA季后赛激战正酣，广东队大胜辽宁队晋级决赛",
        "央行宣布降准0.5个百分点，释放长期资金约1万亿元",
        "原神新版本上线，新角色枫原万叶引发玩家热烈讨论"
    ]
    
    print("\n===== 单文本测试结果 =====")
    for text in test_texts:
        pred_label = predict_single_text(model, tokenizer, text, device)
        print(f"输入文本：{text}")
        print(f"预测标签：{pred_label}\n")
    
    # 也可以让用户交互式输入
    print("===== 交互式测试 =====")
    while True:
        user_input = input("请输入测试文本（输入 'q' 退出）：")
        if user_input.lower() == 'q':
            break
        pred_label = predict_single_text(model, tokenizer, user_input, device)
        print(f"预测标签：{pred_label}\n")
