import argparse
import os
from os.path import join
from sys import path


import warnings

warnings.filterwarnings("ignore")
from transformers import (BertModel, BertTokenizer, RobertaModel,
                          RobertaTokenizer, XLMRobertaModel,
                          XLMRobertaTokenizer, AutoModel, AutoTokenizer)

from dependency import get_dep_matrix_new
from utils import DataLoader, UToken

if __name__ == "__main__":
    MODEL_CLASSES = {
        "bert": (BertModel, BertTokenizer, "bert-base-uncased"),
        "roberta": (RobertaModel, RobertaTokenizer, "roberta-base"),
        "xlmroberta": (XLMRobertaModel, XLMRobertaTokenizer, "xlm-roberta-base"),
        "xlmbert": (BertModel, BertTokenizer, "bert-base-multilingual-cased"),
        "ernie": (AutoModel, AutoTokenizer, "nghuyong/ernie-2.0-en"),
    }
    parser = argparse.ArgumentParser()

    # Model args
    # save_matrix/{model_type}{trained or not}/{dataset}/
    parser.add_argument("--model_path", type=str,default='/your/work/space/save_models/your/model')  # 准备进行masking的模型
    parser.add_argument("--dataset", default="Laptop")
    parser.add_argument("--data_dir", default=r'/your/work/space/RoBERTaABSA/Dataset')
    # Data args
    parser.add_argument("--cuda", default=False, help="invoke to use gpu")
    parser.add_argument(
        "--metric",
        default="dist",
        help="metrics for impact calculation, support [dist, cos] so far",
    )

    args = parser.parse_args()

    if "/" not in args.dataset:
        args.dataset = os.path.join(args.data_dir, args.dataset)
    dataset_name = os.path.basename(args.dataset)
    print(f"Using the following dataset name {dataset_name}")
    if args.model_path.endswith("/"):
        args.model_path = args.model_path[:-1]

    if "/" in args.model_path:
        assert os.path.exists(args.model_path)
        model_type = os.path.basename(args.model_path)
        model_type = model_type.split("-")[0]
    else:
        model_type = args.model_path  # PTMS
    print(model_type)
    model_class, tokenizer_class, pretrained_weights = MODEL_CLASSES[model_type]
    trained_on = ""
    if "/" in args.model_path:
        pretrained_weights = args.model_path
        parts = pretrained_weights.split("/")[-2].split("-")
        if len(parts) == 2:  # "roberta-Laptop"
            trained_on = parts[-1]
            msg = ""
        elif len(parts) == 3:  # "roberta-Laptop-{msg}"
            trained_on = parts[-2]
            msg = parts[-1]
        else:
            raise RuntimeError("Wrong number of parts")

    print(f"Using the following pre-trained weights {pretrained_weights}")
    if model_type != 'ernie':
        model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
        do_lower = False
        tokenizer = tokenizer_class.from_pretrained(MODEL_CLASSES[model_type][2])
    else:
        tokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-2.0-en")
        do_lower = False
        model = BertModel.from_pretrained("nghuyong/ernie-2.0-en", output_hidden_states=True)

    output_dir = "save_matrix/{}{}{}".format(
        model_type, trained_on, "" if "/" not in args.model_path else "trained" + msg
    )
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    print("Folder is {}".format(output_dir))
    args.output_file = output_dir + "/{}-{}.pkl"
    # save_matrix/{model_type}{trained dataset or ""}{""}/{dataset}/{split}-{layer}.pkl

    print(f"Using the following dataset {args.dataset}")
    # exit(0)
    data_bundle = DataLoader().load(args.dataset)
    for name in ["train", "test"]:
        args.data_split = name
        ds = data_bundle.get_dataset(name)
        dataset = []
        for ins in ds:
            line = [
                UToken(
                    tid=0,
                    form="<root>",
                    lemma="[ROOT]",
                    upos="[ROOT]",
                    xpos="[ROOT]",
                    feats="_",
                    head=0,
                    deprel="[ROOT]",
                    deps="_",
                    misc="_",
                    aspects=ins["aspects"],
                )
            ]
            for idx, token in enumerate(ins["tokens"], start=1):
                line.append(
                    UToken(
                        tid=idx,
                        form=token,
                        lemma="[ROOT]",
                        upos="[ROOT]",
                        xpos="[ROOT]",
                        feats="_",
                        head=ins["dep"][idx - 1][1],
                        deprel="[ROOT]",
                        deps="_",
                        misc="_",
                    )
                )
            dataset.append(line)

        out = get_dep_matrix_new(args, model, tokenizer, dataset)
