import torch
from torch.cuda.amp import autocast, GradScaler 
from tqdm import tqdm


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None, accumulation_steps=1):
        """
        Train for one epoch
        """
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        optimizer.zero_grad()

        progress_bar = tqdm(dataloader, desc='Training', leave=False)
        for idx, batch in enumerate(dataloader):
            # Move to device
            if 'input_ids' in batch:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Forward pass with mixed precision
                if scaler:
                    with autocast():
                        outputs = model(input_ids, attention_mask)
                        loss = criterion(outputs, labels) / accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels) / accumulation_steps
                    loss.backward()
            else:
                # For CNN/LSTM models
                inputs = batch['text'].to(device)
                labels = batch['labels'].to(device)

                if scaler:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels) / accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels) / accumulation_steps
                    loss.backward()

            # Gradient accumulation
            if (idx + 1) % accumulation_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            # Statistics
            total_loss += loss.item() * accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return total_loss / len(dataloader), correct / total


def collate_fn_cnn_lstm_fixed(batch, vocab, max_length=256):
    """
    Fixed collate function that handles ReviewDataset properly
    """

    # Extract text and labels from batch items
    # ReviewDataset returns dict with 'text' (str) and 'labels' (tensor)
    texts = []
    labels = []

    for item in batch:
        texts.append(item['text'])  # string
        labels.append(item['labels'])  # already a tensor

    # Stack labels (they are already tensors from __getitem__)
    labels = torch.stack(labels)  # [batch_size]

    # Tokenize and pad texts
    tokenized = []
    for text in texts:
        # Ensure text is string
        if not isinstance(text, str):
            text = str(text)

        tokens = text.lower().split()[:max_length]
        indices = [vocab.word2idx.get(token, vocab.word2idx.get('<unk>', 1)) for token in tokens]

        # Pad or truncate to max_length
        if len(indices) < max_length:
            pad_idx = vocab.word2idx.get('<pad>', 0)
            indices += [pad_idx] * (max_length - len(indices))
        else:
            indices = indices[:max_length]

        tokenized.append(indices)

    text_tensors = torch.tensor(tokenized, dtype=torch.long)

    return {
        'text': text_tensors,
        'labels': labels
    }
