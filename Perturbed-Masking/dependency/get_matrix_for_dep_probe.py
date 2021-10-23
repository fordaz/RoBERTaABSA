from transformers import BertModel, BertTokenizer
import torch
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import os
from .dep_parsing import match_tokenized_to_untokenized


def get_all_subword_id(mapping, idx):
    current_id = mapping[idx]
    id_for_all_subwords = [
        tmp_id for tmp_id, v in enumerate(mapping) if v == current_id
    ]
    return id_for_all_subwords




def get_dep_matrix(args, model, tokenizer, dataset):

    mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    model.eval()

    LAYER = int(args.layers)
    LAYER += 1  # embedding layer
    out = [[] for i in range(LAYER)]
    for line in tqdm(dataset.tokens):
        sentence = [x.form for x in line][1:]

        tokenized_text = tokenizer.tokenize(' '.join(sentence))
        tokenized_text.insert(0, '[CLS]')
        tokenized_text.append('[SEP]')
        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        mapping = match_tokenized_to_untokenized(tokenized_text, sentence)

        # 1. Generate mask indices
        all_layers_matrix_as_list = [[] for i in range(LAYER)]
        for i in range(0, len(tokenized_text)):
            id_for_all_i_tokens = get_all_subword_id(mapping, i)
            tmp_indexed_tokens = list(indexed_tokens)
            for tmp_id in id_for_all_i_tokens:
                if mapping[tmp_id] != -1:  # both CLS and SEP use -1 as id e.g., [-1, 0, 1, 2, ..., -1]
                    tmp_indexed_tokens[tmp_id] = mask_id
            one_batch = [list(tmp_indexed_tokens) for _ in range(0, len(tokenized_text))]
            for j in range(0, len(tokenized_text)):
                id_for_all_j_tokens = get_all_subword_id(mapping, j)
                for tmp_id in id_for_all_j_tokens:
                    if mapping[tmp_id] != -1:
                        one_batch[j][tmp_id] = mask_id

            # 2. Convert one batch to PyTorch tensors
            tokens_tensor = torch.tensor(one_batch)
            segments_tensor = torch.tensor([[0 for _ in one_sent] for one_sent in one_batch])
            if args.cuda:
                tokens_tensor = tokens_tensor.to('cuda')
                segments_tensor = segments_tensor.to('cuda')
                model.to('cuda')

            # 3. get all hidden states for one batch
            with torch.no_grad():
                model_outputs = model(tokens_tensor, segments_tensor)
                # last_layer = model_outputs[0]
                all_layers = model_outputs[-1]  # 12 layers + embedding layer

            # 4. get hidden states for word_i in one batch
            for k, layer in enumerate(all_layers):
                if args.cuda:
                    hidden_states_for_token_i = layer[:, i, :].cpu().numpy()
                else:
                    hidden_states_for_token_i = layer[:, i, :].numpy()
                all_layers_matrix_as_list[k].append(hidden_states_for_token_i)

        for k, one_layer_matrix in enumerate(all_layers_matrix_as_list):
            init_matrix = np.zeros((len(tokenized_text), len(tokenized_text)))
            for i, hidden_states in enumerate(one_layer_matrix):
                base_state = hidden_states[i]
                for j, state in enumerate(hidden_states):
                    if args.metric == 'dist':
                        init_matrix[i][j] = np.linalg.norm(base_state - state)
                    if args.metric == 'cos':
                        init_matrix[i][j] = np.dot(base_state, state) / (
                                    np.linalg.norm(base_state) * np.linalg.norm(state))
            out[k].append((line, tokenized_text, init_matrix))

    for k, one_layer_out in enumerate(out):
        k_output = args.output_file.format(args.model_type, args.metric, args.data_split, str(k))
        with open(k_output, 'wb') as fout:
            pickle.dump(out[k], fout)
            fout.close()


####### revison here
def get_dep_matrix_new(args, model, tokenizer, dataset, output_hidden_states=True):
    
    mask_id = tokenizer.mask_token_id
    model.eval()

    LAYER = 12 + 1 if output_hidden_states else 1
    # LAYER += 1  # embedding layer
    out = [[] for i in range(LAYER)]
    if not isinstance(dataset, list):
        dataset = dataset.tokens
    if args.cuda:
        model.to("cuda")
    with tqdm(total=len(dataset)) as pbar:
        for line in dataset:
            # print(f"Line {line}")
            pbar.update()
            # [x.form for x in line]的结果类似['<root>', "I", "Read", ...]
            sentence = [x.form for x in line][1:]  # 去掉aspect
            # print(f"Sentence {sentence}")
            mapping = []
            indexed_tokens = []

            for idx, word in enumerate(sentence):
                bpes = tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(word, add_special_tokens=True)
                )
                # print(f"Generated {bpes} for word {word}")
                mapping.extend([idx] * len(bpes))  # 记录第idx个word被bpe成几个
                indexed_tokens.extend(bpes)
            mapping.append(-1)
            mapping.insert(0, -1)
            indexed_tokens.append(tokenizer.sep_token_id)
            indexed_tokens.insert(0, tokenizer.cls_token_id)
            # print(f"Indexed tokens {indexed_tokens} {mapping}")

            # 1. Generate mask indices
            all_layers_matrix_as_list = [[] for i in range(LAYER)]
            for i in range(0, len(indexed_tokens)):  # 全体bpe之后的词
                id_for_all_i_tokens = get_all_subword_id(mapping, i)
                tmp_indexed_tokens = list(indexed_tokens)
                for tmp_id in id_for_all_i_tokens:
                    # 把i这个位置的词填充为<MASK>看影响，第一个词是<cls>实际上就是不替换，完整的句子
                    if mapping[tmp_id] != -1:
                        # both CLS and SEP use -1 as id e.g., [-1, 0, 1, 2, ..., -1]
                        tmp_indexed_tokens[tmp_id] = mask_id
                one_batch = [
                    list(tmp_indexed_tokens) for _ in range(0, len(indexed_tokens))
                ]
                for j in range(0, len(indexed_tokens)):
                    id_for_all_j_tokens = get_all_subword_id(mapping, j)
                    for tmp_id in id_for_all_j_tokens:
                        if mapping[tmp_id] != -1:
                            one_batch[j][tmp_id] = mask_id
                # print(f"one_batch {one_batch}")
                # 2. Convert one batch to PyTorch tensors
                tokens_tensor = torch.tensor(one_batch)
                segments_tensor = torch.tensor(
                    [[0 for _ in one_sent] for one_sent in one_batch]
                )
                # print(f"tokens tensor {tokens_tensor} and segments tensor {segments_tensor}")
                if args.cuda:
                    tokens_tensor = tokens_tensor.to("cuda")
                    segments_tensor = segments_tensor.to("cuda")

                # 3. get all hidden states for one batch
                with torch.no_grad():
                    if len(tokens_tensor) > 180:  # 句长大于180
                        print(f"FO-DEBUG getting the layers via method 1")
                        all_layers = []
                        num_segments = len(tokens_tensor) // 50 + int(
                            len(tokens_tensor) % 50 != 0
                        )  # 按50个token一组分组
                        for i in range(num_segments):
                            tokens_tmp1 = tokens_tensor[i * 50 : 50 * (i + 1)]
                            segments_tmp1 = segments_tensor[i * 50 : 50 * (i + 1)]
                            # 这里model的输入是tok+segment,旧版本？
                            # model_output1 = model(tokens_tmp1, segments_tmp1)
                            model_output1 = model(tokens_tmp1)

                            if len(all_layers) == 0:
                                for l in model_output1[-1]:
                                    all_layers.append(l)
                            else:
                                for j, l in enumerate(model_output1[-1]):
                                    all_layers[j] = torch.cat([all_layers[j], l], dim=0)
                    else:
                        # print(f"FO-DEBUG getting the layers via method 2")
                        tmp_model_outputs = model(tokens_tensor, segments_tensor)
                        model_outputs = model(tokens_tensor)
                        # print(f"the reps shape {len(model_outputs)} {model_outputs[0].shape}, {model_outputs[1].shape}")
                        # print(f"resps 2 {type(model_outputs[2])}")
                        # print(f"resps 0 {type(model_outputs[0])}")
                        # all_layers = model_outputs[-1]  # 12 layers + embedding layer
                        all_layers = model_outputs[-1] if output_hidden_states else [model_outputs[0]]

                # 4. get hidden states for word_i in one batch
                # print(f"FO-DEBUG all_layers {type(all_layers)} {len(all_layers)}")
                # exit(1)
                for k, layer in enumerate(all_layers):
                    # print(f"    FO-DEBUG all_layer {type(layer)} {layer.shape}")
                    # exit(0)
                    if args.cuda:
                        hidden_states_for_token_i = layer[:, i, :].cpu().numpy()
                        #  if output_hidden_states else layer[i, :].cpu().numpy()
                    else:
                        hidden_states_for_token_i = layer[:, i, :].numpy()
                        #  if output_hidden_states else layer[i, :].numpy()
                    all_layers_matrix_as_list[k].append(hidden_states_for_token_i)
            # all_layers_matrix_as_list: N_layer x len(sub_token_length) x len(sub_token_length) x hidden_size
            # print(f"all_layers_matrix_as_list {len(all_layers_matrix_as_list)} {len(all_layers_matrix_as_list[0])} {type(all_layers_matrix_as_list[0][0])} {all_layers_matrix_as_list[0][0].shape}")
            # exit(0)
            for k, one_layer_matrix in enumerate(all_layers_matrix_as_list):
                # print(f"Processing layer {k} {one_layer_matrix}")
                # one_layer_matrix: len(sub_token_length) x len(sub_token_length) x hidden_size
                init_matrix = np.zeros((len(indexed_tokens), len(indexed_tokens)))
                for i, hidden_states in enumerate(one_layer_matrix):
                    # print(f" Processing hidden states {i} {hidden_states}")
                    # hidden_states: len(sub_token_length) x hidden_size当i被mask掉时，剩下的位置依次被mask的表示
                    base_state = hidden_states[i]  # 自己位置，这是由于上面两次mask都到同一个[MASK]位置了
                    # print(f"  Processing base state {i} {base_state}")
                    for j, state in enumerate(hidden_states):
                        # print(f"   Processing state {j} {state}")
                        if args.metric == "dist":
                            init_matrix[i][j] = np.linalg.norm(base_state - state)
                        if args.metric == "cos":
                            init_matrix[i][j] = np.dot(base_state, state) / (
                                np.linalg.norm(base_state) * np.linalg.norm(state)
                            )
                # print(f"Mapping {mapping} {init_matrix}")
                # exit(1)
                out[k].append((line, mapping, init_matrix))

    if hasattr(args, "output_file") and args.output_file != "":
        for k, one_layer_out in enumerate(out):
            k_output = args.output_file.format(args.data_split, str(k))
            with open(k_output, "wb") as fout:
                pickle.dump(out[k], fout)
                fout.close()
    return out
