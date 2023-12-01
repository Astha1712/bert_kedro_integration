from pandas import DataFrame
from transformers import BertTokenizer, BertForMaskedLM
import torch

def bert_language_model(input_data: DataFrame) -> any:
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    # Tokenize input text column
    tokenized_text = input_data['input_text'].apply(lambda x: tokenizer.encode(x, return_tensors='pt'))

    # Get model predictions
    with torch.no_grad():
        predictions = [model(tokens) for tokens in tokenized_text]

    return predictions
