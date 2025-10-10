import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

class Vocabulary:
    """Fixed vocabulary class with consistent token naming"""

    def __init__(self, max_size=50000, min_freq=2):
        self.max_size = max_size
        self.min_freq = min_freq
        # Use consistent lowercase tokens
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}

    def build_vocab(self, texts):
        from collections import Counter
        from tqdm import tqdm

        print("[INFO] Building vocabulary...")
        word_counts = Counter()

        for text in tqdm(texts, desc='Counting words'):
            words = str(text).lower().split()
            word_counts.update(words)

        # Filter by frequency and take top max_size
        valid_words = [(w, c) for w, c in word_counts.items() if c >= self.min_freq]
        valid_words.sort(key=lambda x: x[1], reverse=True)
        valid_words = valid_words[:self.max_size - 2]  # Reserve space for pad and unk

        # Add to vocabulary
        for idx, (word, count) in enumerate(valid_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"[INFO] Vocabulary size: {len(self.word2idx)}")
        print(f"[INFO] Most common words: {valid_words[:10]}")

    def encode(self, text, max_length):
        """Encode text to indices"""
        words = str(text).lower().split()[:max_length]
        # Use lowercase '<unk>' consistently
        indices = [self.word2idx.get(w, self.word2idx['<unk>']) for w in words]

        # Pad or truncate
        if len(indices) < max_length:
            indices += [self.word2idx['<pad>']] * (max_length - len(indices))
        else:
            indices = indices[:max_length]

        return indices


class ReviewDataset(Dataset):
    """Fixed PyTorch Dataset for drug reviews"""

    def __init__(self, texts, labels, tokenizer=None, max_length=256,
                 augment=False, augmentation_prob=0.1):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.augmentation_prob = augmentation_prob

    def __len__(self):
        return len(self.texts)

    def augment_text(self, text):
        """Simple text augmentation by random word dropout"""
        import numpy as np

        if not self.augment or np.random.random() > self.augmentation_prob:
            return text

        words = text.split()
        # Randomly drop 10% of words
        if len(words) > 10:
            num_drops = max(1, int(len(words) * 0.1))
            drop_indices = np.random.choice(len(words), num_drops, replace=False)
            words = [w for i, w in enumerate(words) if i not in drop_indices]
        return ' '.join(words)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Apply augmentation during training
        if self.augment and self.tokenizer is None:  # Only for CNN/LSTM
            text = self.augment_text(text)

        label = self.labels[idx]

        if self.tokenizer:
            # For BERT models
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            # For CNN/LSTM models - return raw text
            return {
                'text': text,
                'labels': torch.tensor(label, dtype=torch.long)
            }


class ImprovedTextCNN(nn.Module):
    """
    Improved Text-CNN with enhanced regularization
    """

    def __init__(self, vocab_size, embed_dim=300, num_classes=3,
                 kernel_sizes=[2, 3, 4, 5], num_filters=128,
                 dropout=0.5, embed_dropout=0.3, l2_reg=0.01,
                 use_batch_norm=True, use_residual=True):
        super(ImprovedTextCNN, self).__init__()


        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.l2_reg = l2_reg
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual

        # Embedding layer with padding_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(embed_dropout)

        # Multiple convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])

        # Batch normalization for each conv layer
        if use_batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(num_filters) for _ in kernel_sizes
            ])

        # Residual connection (if input dim matches)
        if use_residual and embed_dim == num_filters:
            self.residual_conv = nn.Conv1d(embed_dim, num_filters, kernel_size=1)

        # Attention mechanism for feature importance
        total_filters = num_filters * len(kernel_sizes)
        self.attention = nn.Sequential(
            nn.Linear(total_filters, total_filters // 2),
            nn.Tanh(),
            nn.Linear(total_filters // 2, total_filters),
            nn.Sigmoid()
        )

        # Highway network for better gradient flow
        self.highway = nn.Sequential(
            nn.Linear(total_filters, total_filters),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.highway_gate = nn.Sequential(
            nn.Linear(total_filters, total_filters),
            nn.Sigmoid()
        )

        # Classification layers with deeper architecture
        self.fc = nn.Sequential(
            nn.Linear(total_filters, total_filters // 2),
            nn.ReLU(),
            nn.BatchNorm1d(total_filters // 2) if use_batch_norm else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(total_filters // 2, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better convergence"""
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0)

        for module in self.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len]
        Returns:
            logits: [batch_size, num_classes]
        """
        # Embedding: [batch, seq_len, embed_dim]
        embedded = self.embedding(x)
        embedded = self.embed_dropout(embedded)

        # Transpose for Conv1d: [batch, embed_dim, seq_len]
        embedded = embedded.transpose(1, 2)

        # Apply multiple conv layers
        conv_outputs = []
        for i, conv in enumerate(self.convs):
            # Convolution
            conv_out = conv(embedded)

            # Batch normalization
            if self.use_batch_norm:
                conv_out = self.batch_norms[i](conv_out)

            # Activation
            conv_out = F.relu(conv_out)

            # Max pooling over time
            pooled = F.max_pool1d(conv_out, conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))

        # Concatenate all conv outputs
        concatenated = torch.cat(conv_outputs, dim=1)

        # Attention mechanism
        attention_weights = self.attention(concatenated)
        attended = concatenated * attention_weights

        # Highway network
        highway_out = self.highway(attended)
        gate = self.highway_gate(attended)
        combined = gate * highway_out + (1 - gate) * attended

        # Classification
        logits = self.fc(combined)

        return logits

    def get_l2_loss(self):
        """Calculate L2 regularization loss"""
        l2_loss = 0
        for conv in self.convs:
            l2_loss += torch.norm(conv.weight, p=2)
        return self.l2_reg * l2_loss


class ImprovedBiLSTMAttention(nn.Module):
    """
    Improved Bi-LSTM with attention - Fixed version
    """

    def __init__(self, vocab_size, embed_dim=300, hidden_dim=128,
                 num_classes=3, num_layers=2, dropout=0.5,
                 recurrent_dropout=0.5, embed_dropout=0.4, l2_reg=0.02):
        super(ImprovedBiLSTMAttention, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.embed_ln = nn.LayerNorm(embed_dim)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                           bidirectional=True, batch_first=True,
                           dropout=recurrent_dropout if num_layers > 1 else 0)

        self.num_heads = 4
        self.attention_dim = hidden_dim * 2
        self.multihead_attention = nn.MultiheadAttention(
            self.attention_dim, self.num_heads, dropout=dropout, batch_first=True
        )

        self.attention_ln = nn.LayerNorm(self.attention_dim)
        self.lstm_ln = nn.LayerNorm(self.attention_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 0.8)
        self.dropout3 = nn.Dropout(dropout * 0.6)

        hidden_fc = hidden_dim
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_fc)
        self.fc1_ln = nn.LayerNorm(hidden_fc)
        self.fc2 = nn.Linear(hidden_fc, num_classes)

        self.l2_reg = l2_reg
        self.max_grad_norm = 0.5

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.embedding(x)
        x = self.embed_dropout(x)
        x = self.embed_ln(x)

        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_ln(lstm_out)
        lstm_out = self.dropout3(lstm_out)

        attn_out, _ = self.multihead_attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attention_ln(attn_out)

        avg_pool = torch.mean(attn_out, dim=1)
        max_pool = torch.max(attn_out, dim=1)[0]
        combined = (avg_pool + max_pool) / 2

        out = self.dropout1(combined)
        out = F.relu(self.fc1(out))
        out = self.fc1_ln(out)
        out = self.dropout2(out)
        out = self.fc2(out)

        return out

    def get_l2_loss(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, 2)
        return self.l2_reg * l2_loss


class ImprovedBERTClassifier(nn.Module):
    """
    Improved BERT classifier - Fixed version
    """

    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT',
                 num_classes=3, dropout=0.3, hidden_dropout=0.5,
                 freeze_bert_layers=4, use_pooler=True):
        super(ImprovedBERTClassifier, self).__init__()

        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        self.use_pooler = use_pooler

        if freeze_bert_layers > 0:
            for layer in self.bert.encoder.layer[:freeze_bert_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.ln = nn.LayerNorm(self.bert.config.hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(hidden_dropout)

        hidden_dim = self.bert.config.hidden_size // 2
        self.fc1 = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        self.label_smoothing = 0.1

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if self.use_pooler:
            pooled_output = outputs.pooler_output
        else:
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled_output = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        pooled_output = self.ln(pooled_output)
        pooled_output = self.dropout1(pooled_output)

        hidden = F.relu(self.fc1(pooled_output))
        hidden = self.dropout2(hidden)
        logits = self.fc2(hidden)

        return logits
