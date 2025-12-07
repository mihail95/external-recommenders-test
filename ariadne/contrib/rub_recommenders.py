from pathlib import Path

from cassis import Cas
import more_itertools as mit
from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_prediction, TOKEN_TYPE, SENTENCE_TYPE
from transformers import AutoModelForTokenClassification, AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
import json
import re
import csv

PUNCT_RE = re.compile(r"^(.+?)([.,!?;:])$")

# Choose device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

MODEL_PATH = "saved-models/bert-base-multilingual-cased_condensed_sents_reddit_sermon_train"

print("=== Loading tokenizer/model at import time ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

id2label = {
    0: "NON_METAPHOR",
    1: "METAPHOR",
}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_PATH,
    id2label=id2label,
    label2id=label2id,
)
model.to(DEVICE)
model.eval()

LABEL_LIST = [id2label[i] for i in range(len(id2label))]
print("Model labels:", LABEL_LIST)

print("=== Model loaded ===")

print("torch.cuda.is_available():", torch.cuda.is_available())
print("Model on device:", next(model.parameters()).device)


def sentences_to_df(sentences):
    tokens = []
    labels = []

    for sent in sentences:
        if not sent:
            continue

        for tok in sent:
            tokens.append(tok)
            labels.append("l")  # placeholder, not used for inference

        # sentence separator = blank line -> NaN in both columns
        tokens.append(np.nan)
        labels.append(np.nan)

    return pd.DataFrame({"token": tokens, "label": labels})

def split_punct(token: str):
    m = PUNCT_RE.match(token)
    if not m:
        return [token]

    text, punct = m.groups()

    # If somehow the token is only punctuation, don't split
    if text.strip() == "":
        return [token]

    return [text, punct]


class MetaphorClassifier(Classifier):
    def predict(self, cas: Cas, layer: str, feature: str,
                project_id: str, document_id: str, user_id: str):

        print("\nReceived a request with the following parameters:")
        print(f"layer: {layer}")
        print(f"feature: {feature}")
        print(f"project_id: {project_id}")
        print(f"document_id: {document_id}")
        print(f"user_id: {user_id}")

        if feature != "Metaphor":
            print("Feature is not 'Metaphor' - skipping.")
            return
        
        sentences_tokens = []
        sentences_offsets = []

        for sentence in cas.select(SENTENCE_TYPE):
            cas_tokens = list(cas.select_covered(TOKEN_TYPE, sentence))

            sent_tokens = []
            sent_offsets = []

            for tok in cas_tokens:
                text = tok.get_covered_text()
                sent_tokens.append(text)
                sent_offsets.append((tok.begin, tok.end))

            sentences_tokens.append(sent_tokens)
            sentences_offsets.append(sent_offsets)

        if not sentences_tokens:
            print("No sentences found in CAS.")
            return

        sentence_label_seqs, sentence_conf_seqs = predict_labels_for_sentences(sentences_tokens)
        print("Got predictions for", len(sentence_label_seqs), "sentences")

        for sent_offsets, labels, confs in zip(sentences_offsets, sentence_label_seqs, sentence_conf_seqs):
            for (begin, end), label, conf in zip(sent_offsets, labels, confs):
                if label != "METAPHOR":
                    continue

                prediction = create_prediction(
                    cas,
                    layer,
                    feature,
                    begin,
                    end,
                    label,
                    score=float(conf) 
                )

                cas.add_annotation(prediction)

def load_model_and_make_predictions(saved_model_path, test_data, output_file):

    global tokenizer, model, LABEL_LIST

    # Build inference dataset from df
    test_dataset = build_inference_dataset(test_data, tokenizer)

    # Prepare dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    # For saving outputs
    all_tokens = []
    all_predictions = []
    all_true_labels = []
    all_tokens_remerged = []

    bcounter = 0
    seen = set()  # to avoid duplicate tokens because of strides

    for batch in test_dataloader:
        bcounter += 1

        inputs = {}
        for k, v in batch.items():
            if k in ["input_ids", "attention_mask", "labels"]:
                inputs[k] = v.to(DEVICE)
            else:
                # keep word_ids on CPU - we only use them for indexing later
                inputs[k] = v

        # Debug shapes/dtypes (keep this while debugging)
        print("=== BATCH ===")
        for k, v in inputs.items():
            print(k, v.shape, v.dtype)
        print("Model device:", next(model.parameters()).device)
        print("Model dtype:", next(model.parameters()).dtype)

        # Safe forward to see real errors
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                logits = outputs.logits
        except Exception as e:
            import traceback
            print("=== EXCEPTION IN MODEL FORWARD ===")
            traceback.print_exc()
            print("Error type:", type(e))
            print("Error message:", e)
            raise

        predictions = torch.argmax(logits, dim=-1).cpu().numpy()

        labels = inputs["labels"].cpu().numpy()
        input_ids = inputs["input_ids"].cpu().numpy()
        attention_mask = inputs["attention_mask"].cpu().numpy()
        word_ids = inputs["word_ids"].cpu().numpy()

        for pred, label, input_id_seq, attention, word_id_seq in zip(
            predictions, labels, input_ids, attention_mask, word_ids
        ):
            # tokens:
            tokens = tokenizer.convert_ids_to_tokens(input_id_seq)

            tokens_remerged = []
            tok_merged = ""

            # with overlap (strides)
            wo_tokens_clean = []
            wo_pred_labels = []
            wo_true_labels = []

            for i, token in enumerate(tokens):
                w_id = word_id_seq[i]

                if label[i] != -100:
                    # keep only positions corresponding to real words
                    wo_tokens_clean.append((token, w_id))

                    # predicted label from MODEL labels
                    wo_pred_labels.append(LABEL_LIST[pred[i]])

                    # "true" label – here all dummy, but we mirror the structure
                    wo_true_labels.append(LABEL_LIST[label[i]])

                    # new real token found -> close previous merged token
                    if tok_merged:
                        tokens_remerged.append(tok_merged)
                    tok_merged = token.strip("▁")

                else:
                    # special/padding/subword
                    if i == 0:
                        continue

                    # subword of current token
                    if attention[i] != 0:
                        if token not in ["</s>"]:
                            tok_merged += token
                    else:
                        # padding: close merged token
                        if tok_merged:
                            tokens_remerged.append(tok_merged)
                        tok_merged = ""

                # if we reached the last token: append to list
                if i == len(tokens) - 1:
                    if tok_merged:
                        tokens_remerged.append(tok_merged)

            tokens_clean = []
            pred_labels = []
            true_labels = []
            final_tokens_remerged = []

            # remove overlapping tokens from stride
            for tok, pred_lab, true_lab, merged in zip(
                wo_tokens_clean, wo_pred_labels, wo_true_labels, tokens_remerged
            ):
                # empty set of seen tokens when word_id starts with 0 again
                if tok[1] == 0:
                    seen = set()
                if tok[1] not in seen:
                    tokens_clean.append(tok[0])
                    pred_labels.append(pred_lab)
                    true_labels.append(true_lab)
                    final_tokens_remerged.append(merged)
                    seen.add(tok[1])

            all_tokens.append(tokens_clean)
            all_predictions.append(pred_labels)
            all_true_labels.append(true_labels)
            all_tokens_remerged.append(final_tokens_remerged)

    # Flatten or format rows to make them CSV-friendly
    rows = []
    for tokens, preds, trues, tokens_remerged in zip(
        all_tokens, all_predictions, all_true_labels, all_tokens_remerged
    ):
        for token, pred, true, token_remerged in zip(
            tokens, preds, trues, tokens_remerged
        ):
            rows.append(
                {
                    "token": token,
                    "token_full": token_remerged,
                    "predicted_label": pred,
                    "true_label": true,
                }
            )
        rows.append(
            {"token": "", "token_full": "", "predicted_label": "", "true_label": ""}
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, sep="\t")
    print(f"Predictions saved to {output_file}")

def format_for_TokenClf(df) -> list:
    data_list = []
    sentence = []
    labels = []
    for index, row in df.iterrows():
        if row.isnull().all():
            if not sentence: continue #may happen if there are multiple empty lines in a row, then ignore them
            data_list.append((sentence, labels))
            sentence = []
            labels = []
            continue
        sentence.append(str(row["token"]))
        labels.append(str(row["label"]))

        if index == len(df) - 1:
            data_list.append((sentence, labels))

    return data_list

class OurDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def tokenize_and_align_labels(texts, tags, tokenizer, label_list):

    label_to_id = {l: i for i, l in enumerate(label_list)}
    
    
    tokenized_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
        # the following are for overlapping windows if the sequence is too long
        stride=32,
        return_overflowing_tokens=True,
        return_offsets_mapping=True
    )
    
    overflow_to_sample_mapping = tokenized_inputs["overflow_to_sample_mapping"]
    labels = []
    all_word_ids = []

    for i in range(len(tokenized_inputs["input_ids"])):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        sample_idx = overflow_to_sample_mapping[i]   # map back to original example
        label = tags[sample_idx]                     # correct label sequence
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                if word_idx < len(label):  # safeguard
                    label_ids.append(label_to_id[label[word_idx]])
                else:
                    label_ids.append(-100)  # safeguard fallback
            # For the other tokens in a word, we set the label to -100
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
        all_word_ids.append(word_ids)

    tokenized_inputs["labels"] = labels
    word_ids = [[-1 if w is None else w for w in encoding.word_ids] for encoding in tokenized_inputs.encodings]
    tokenized_inputs["word_ids"] = word_ids
    
    return tokenized_inputs

def build_dataset(df, model_for_tokenization):

    tokenizer = AutoTokenizer.from_pretrained(model_for_tokenization)

    data_tuplelist = format_for_TokenClf(df)
    tags=[tup[1] for tup in data_tuplelist]
    texts=[tup[0] for tup in data_tuplelist]
    label_list = sorted(set([tag for liste in tags for tag in liste]))
    input_and_labels = tokenize_and_align_labels(texts, tags, tokenizer, label_list)
    torch_dataset = OurDataset(input_and_labels, input_and_labels["labels"])

    return torch_dataset, label_list

def build_inference_dataset(df, tokenizer):
    """
    df: DataFrame with columns 'token' and 'label' (label is ignored semantically).
    Returns: OurDataset with input_ids, attention_mask, labels (mask), word_ids.
    """
    # Reuse your existing formatting: df -> list of (sentence_tokens, sentence_labels)
    data_tuplelist = format_for_TokenClf(df)

    # Make absolutely sure we have list[list[str]]
    texts = []
    for sent, _labels in data_tuplelist:
        if isinstance(sent, (list, tuple)):
            texts.append([str(tok) for tok in sent])
        else:
            # just in case something weird sneaks in
            texts.append([str(sent)])

    if not texts:
        raise ValueError("build_inference_dataset: got no sentences from df")

    print("build_inference_dataset: #sentences:", len(texts))
    print("build_inference_dataset: first sentence:", texts[0][:10])

    # Dummy labels: one label per word (we only need them to mark real tokens vs specials)
    dummy_label = "l"
    tags = [[dummy_label] * len(sent) for sent in texts]
    label_list = [dummy_label]

    # This will create:
    # - input_ids
    # - attention_mask
    # - labels (0 / -100, we use as mask)
    # - word_ids
    tokenized_inputs = tokenize_and_align_labels(texts, tags, tokenizer, label_list)

    torch_dataset = OurDataset(tokenized_inputs, tokenized_inputs["labels"])
    return torch_dataset

def predict_labels_for_sentences(sentences):
    global tokenizer, model, LABEL_LIST

    # Tokenize as pre-tokenized input
    tokenized = tokenizer(
        sentences,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        return_attention_mask=True,
    )

    # Turn into tensors and move to device
    input_ids = torch.tensor(tokenized["input_ids"]).to(model.device)
    attention_mask = torch.tensor(tokenized["attention_mask"]).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch, seq, num_labels]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_ids = logits.argmax(dim=-1).cpu().numpy()

    sentence_label_seqs = []
    sentence_conf_seqs = []

    for sent_idx, sent_tokens in enumerate(sentences):
        word_ids = tokenized.word_ids(batch_index=sent_idx)
        seq_preds = pred_ids[sent_idx]
        seq_probs = probs[sent_idx]

        word_labels = [None] * len(sent_tokens)
        word_confs = [0.0] * len(sent_tokens)

        for pos, w_id in enumerate(word_ids):
            if w_id is None:
                continue
            # first subword of a word
            if word_labels[w_id] is None:
                label_idx = int(seq_preds[pos])
                word_labels[w_id] = LABEL_LIST[label_idx]
                word_confs[w_id] = float(seq_probs[pos, label_idx])

        # safety fallback
        for i, lbl in enumerate(word_labels):
            if lbl is None:
                word_labels[i] = "NON_METAPHOR"
                word_confs[i] = 0.0

        sentence_label_seqs.append(word_labels)
        sentence_conf_seqs.append(word_confs)

    return sentence_label_seqs, sentence_conf_seqs