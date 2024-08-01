import torch
from transformers import BertTokenizerFast, BertModel
import numpy as np
import joblib

model_name = 'kykim/bert-kor-base'
Tokenizer = BertTokenizerFast.from_pretrained(model_name, local_files_only=True)
Model = BertModel.from_pretrained(model_name, local_files_only=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Model = Model.to(device)  # 모델을 GPU로 이동

def sentence2vector(sentence):
    tokens = Tokenizer(
        sentence, 
        return_tensors='pt',
        truncation=True, 
        padding='max_length',
        add_special_tokens=True,
        max_length=256, 
    )
    # 토큰을 GPU로 이동
    tokens = {k: v.to(device) for k, v in tokens.items()}
    
    with torch.no_grad():  # 그래디언트 계산 비활성화
        outputs = Model(**tokens)
    
    # 임베딩 추출 및 CPU로 이동
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embedding

def classify(sentence):
    vector = sentence2vector(sentence)
    model = joblib.load('logistic_regression_model.pkl')
    class_probabilities = model.predict_proba(vector)[0]
    # 가장 높은 확률을 가진 클래스의 확률 반환
    max_probability = np.max(class_probabilities)
    return max_probability > 0.8
