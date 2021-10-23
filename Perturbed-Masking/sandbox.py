import os
# import pandas as pd
import torch
from transformers import BertModel, BertTokenizer, AutoTokenizer
from transformers import RobertaModel, RobertaTokenizer, AutoModel

def testing_bert():
    bert_weights_name = 'bert-base-uncased'
    bert_tokenizer = BertTokenizer.from_pretrained(bert_weights_name)
    bert_model = BertModel.from_pretrained(bert_weights_name, output_hidden_states=True)

    example_text = "Bert knows Snuffleupagus"

    ex_ids = bert_tokenizer.encode(example_text, add_special_tokens=True)

    with torch.no_grad():
        reps = bert_model(torch.tensor([ex_ids]))
        print(f"bert response {len(reps)}")
        # print(type(reps[0]), type(reps[1]))
        print(f"the reps shape {reps[0].shape}, {reps[1].shape} {type(reps[2])} {len(reps[2])}")
    
def testing_ernie():
    ernie_tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-en")
    ernie_model = AutoModel.from_pretrained("nghuyong/ernie-2.0-en", output_hidden_states=True)

    example_text = "Bert knows Snuffleupagus"

    ex_ids = ernie_tokenizer.encode(example_text, add_special_tokens=True)

    with torch.no_grad():
        reps = ernie_model(torch.tensor([ex_ids]))
        print(f"bert response {len(reps)}")
        # print(type(reps[0]), type(reps[1]))
        if len(reps) > 2:
            print(f"the reps shape {reps[0].shape}, {reps[1].shape} {type(reps[2])} {len(reps[2])}")
        else:
            print(f"the reps shape {reps[0].shape}, {reps[1].shape}")


def testing_ernie_2():
    MODEL_CLASSES = {
        "bert": (BertModel, BertTokenizer, "bert-base-uncased"),
        "roberta": (RobertaModel, RobertaTokenizer, "roberta-base"),
        "xlmbert": (BertModel, BertTokenizer, "bert-base-multilingual-cased"),
        "ernie": (AutoModel, AutoTokenizer, "nghuyong/ernie-2.0-en"),
    }

    model_type = "ernie"
    model_class, tokenizer_class, pretrained_weights = MODEL_CLASSES[model_type]
    print(f"Using the following pre-trained weights {pretrained_weights}")
    model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
    tokenizer = tokenizer_class.from_pretrained(MODEL_CLASSES[model_type][2])

testing_ernie()
# testing_bert()