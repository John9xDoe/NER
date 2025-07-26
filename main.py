import logging
import random

def load_data(filepath):
    tokens, labels = [], []
    ex_tokens, ex_labels = [], []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:
                if ex_tokens:
                    tokens.append(ex_tokens)
                    labels.append(ex_labels)
                    ex_tokens, ex_labels = [], []
                continue
            try:
                token, label = line.split()
                ex_tokens.append(token)
                ex_labels.append(label)
            except ValueError as e:
                logging.error(e)

    if ex_tokens:
        tokens.append(ex_tokens)
        labels.append(ex_labels)

    return tokens, labels

def train_test_split(tokens, labels, test_size=0.1, shuffle=False):
    if shuffle:
        combine = list(zip(tokens, labels))
        random.shuffle(combine)
        tokens, labels = zip(*combine)

    split_idx = int((1 - test_size) * len(tokens))

    train_tokens, test_tokens = tokens[:split_idx], tokens[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]

    return train_tokens, test_tokens, train_labels, test_labels

def pad_batch(batch_tokens, batch_labels, pad_id='PAD', ignore_id=-100):
    max_len = max(len(seq) for seq in batch_tokens)

    padded_tokens = []
    padded_labels = []
    mask = []

    for tokens, labels in zip(batch_tokens, batch_labels):
        pad_len = max_len - len(tokens)

        padded_tokens.append(tokens + pad_len * [pad_id])
        padded_labels.append(labels + pad_len * [ignore_id])
        mask.append([1] * len(tokens) + [0] * pad_len)

    return padded_tokens, padded_labels, mask

def encode(tokens, labels, token2id, label2id, unk_tok='<UNK>'):
    input_ids = [token2id.get(tok, token2id[unk_tok]) for tok in tokens]
    label_ids = [label2id[lab] for lab in labels]
    return input_ids, label_ids

def make_batches(encoded_data, batch_size):
    for i in range(0, len(encoded_data), batch_size):
        batch = encoded_data[i: i + batch_size]
        batch_inputs, target_inputs = zip(*batch)

        padded_token, padded_labels, mask = pad_batch(batch_inputs, target_inputs)
        yield padded_token, padded_labels, mask

def start():
    tokens, labels = load_data('test_dataset.txt')

    all_tokens = [token for sentence in tokens for token in sentence]
    all_labels = [label for sentence in labels for label in sentence]

    vocab = ['PAD', '<UNK>'] + sorted(set(all_tokens))
    token2id = {token: idx for idx, token in enumerate(vocab)}
    id2token = {idx: token for token, idx in token2id.items()}

    unique_labels = sorted(set(all_labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    train_tokens, test_tokens, train_labels, test_labels = train_test_split(tokens, labels, test_size=0.1, shuffle=True)

    train_encoded = [encode(toks, labs, token2id, label2id) for toks, labs in zip(train_tokens, train_labels)]

    for idx, (inputs, labels, mask) in enumerate(make_batches(train_encoded, batch_size=32)):
        print(f"{idx}: {inputs} | {labels} | {mask}")

if __name__ == '__main__':
    start()