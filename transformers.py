import math
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len=512):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x)
        x += self.positional_encoding[:, :seq_len, :]
        return x

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        self.out_linear = nn.Linear(embed_dim, embed_dim)
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        return self.out_linear(attention_output)

class AddNorm(nn.Module):
    def __init__(self, embed_dim):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x, sublayer_output):
        return self.norm(x + sublayer_output)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim=2048):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dim, embed_dim)
    def forward(self, x):
        return self.linear_2(self.relu(self.linear_1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim=2048):
        super(EncoderLayer, self).__init__()
        self.self_attention = SelfAttention(embed_dim, num_heads)
        self.add_norm1 = AddNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, hidden_dim)
        self.add_norm2 = AddNorm(embed_dim)
    def forward(self, x):
        attention_output = self.self_attention(x)
        x = self.add_norm1(x, attention_output)
        ffn_output = self.feed_forward(x)
        x = self.add_norm2(x, ffn_output)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim=2048, max_len=512):
        super(TransformerEncoder, self).__init__()
        self.embedding = Embeddings(vocab_size, embed_dim, max_len=max_len)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return x

class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MaskedSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        self.out_linear = nn.Linear(embed_dim, embed_dim)
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)

        scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_linear(attention_output)

class EncoderDecoderAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(EncoderDecoderAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        self.out_linear = nn.Linear(embed_dim, embed_dim)
    def forward(self, x, encoder_outpupt):
        batch_size, seq_len, _ = x.size()
        enc_seq_len = encoder_outpupt.size(1)

        Q = self.q_linear(x)
        K = self.k_linear(encoder_outpupt)
        V = self.v_linear(encoder_outpupt)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, enc_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, enc_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)

        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_linear(attention_output)

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim=2048):
        super(DecoderLayer, self).__init__()
        self.masked_self_attention = MaskedSelfAttention(embed_dim, num_heads)
        self.add_norm1 = AddNorm(embed_dim)
        self.encoder_decoder_attention = EncoderDecoderAttention(embed_dim, num_heads)
        self.add_norm2 = AddNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, hidden_dim)
        self.add_norm3 = AddNorm(embed_dim)
    def forward(self, x, encoder_outupt):
        masked_attention_output = self.masked_self_attention(x)
        x = self.add_norm1(x, masked_attention_output)
        enc_dec_attention_output = self.encoder_decoder_attention(x, encoder_outupt)
        x = self.add_norm2(x, enc_dec_attention_output)
        ffn_output = self.feed_forward(x)
        x = self.add_norm3(x, ffn_output)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim=2048, max_len=512):
        super(TransformerDecoder, self).__init__()
        self.embedding = Embeddings(vocab_size, embed_dim, max_len=max_len)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])
    def forward(self, x, encoder_outupt):
        x = self.embedding(x)
        for layer in self.decoder_layers:
            x = layer(x, encoder_outupt)
        return x

class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
    def build_vocab(self, sentences):
        unique_word = set(word for sentence in sentences for word in sentence.split())
        self.word2idx = {word: idx+4 for idx, word in enumerate(unique_word)}
        self.word2idx["<pad>"] = 0
        self.word2idx["<sos>"] = 1
        self.word2idx["<eos>"] = 2
        self.word2idx["<unk>"] = 3
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
    def encode(self, sentence):
        return [self.word2idx.get(word, self.word2idx["<unk>"]) for word in sentence.split()]
    def decode(self, indices):
        return " ".join([self.idx2word.get(idx, "<unk>") for idx in indices])

class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data
    def __len__(self):
        return len(self.src_data)
    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx][:-1], self.tgt_data[idx][1:]

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_len):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(src_vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_len)
        self.decoder = TransformerDecoder(tgt_vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_len)
        self.output_layer = nn.Linear(embed_dim, tgt_vocab_size)
        self.device = device
    def forward(self, src, tgt):
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        encoder_output = self.encoder(src)
        decoder_output = self.decoder(tgt, encoder_output)
        return self.output_layer(decoder_output)

def pad_sequence(seq, max_len, pad_idx=0):
    return seq + [pad_idx] * (max_len - len(seq))

def get_max_len(data, eng_tokenizer, span_tokenizer):
    max_len = 0
    for eng, spa in data:
        eng_tokens = [1] + eng_tokenizer.encode(eng) + [2]
        spa_tokens = [1] + span_tokenizer.encode(spa) + [2]
        max_len = max(max_len, len(eng_tokens), len(spa_tokens))
    return max_len

def preprocess_data(data, max_len, eng_tokenizer, span_tokenizer, device):
    english_data = []
    spanish_data = []

    for eng, spa in data:
        eng_tokens = [1] + eng_tokenizer.encode(eng) + [2]
        spa_tokens = [1] + span_tokenizer.encode(spa) + [2]

        eng_tokens = pad_sequence(eng_tokens, max_len)
        spa_tokens = pad_sequence(spa_tokens, max_len)

        english_data.append(eng_tokens)
        spanish_data.append(spa_tokens)
    print(f'Data Preprocessing Done! with Max Length: {max_len}\n')
    return torch.tensor(english_data).to(device), torch.tensor(spanish_data).to(device)

def train(model, dataloader, criterion, optimizer, device, num_epochs):
    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_loss = 0
        for src, tgt_input, tgt_output in tqdm(dataloader):
            optimizer.zero_grad()

            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)

            outputs = model(src, tgt_input)
            loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_output.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'\nEpoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss/len(dataloader):.4f}')
    return model

def tokenize_input(sentence, tokenizer, max_len):
    tokens = tokenizer.encode(sentence)
    tokens = [1] + tokens + [2]
    tokens = pad_sequence(tokens, max_len)
    return torch.tensor(tokens).unsqueeze(0)

def test(model, input_sentence, english_tokenizer, spanish_tokenizer, device, max_length):
    model.eval()
    src_tensor = tokenize_input(input_sentence, english_tokenizer, max_length).to(device)

    with torch.no_grad():
        encoder_output = model.encoder(src_tensor)
        tgt_tokens = torch.tensor([[1]]).to(device)
        translated_tokens = []
        for _ in range(max_length):
            with torch.no_grad():
                output = model.decoder(tgt_tokens, encoder_output)
                output = model.output_layer(output)
            next_token = output[:, -1, :].argmax(1).item()
            if next_token == 2:
                break
            translated_tokens.append(next_token)
            tgt_tokens = torch.cat([tgt_tokens, torch.tensor([[next_token]]).to(device)], dim=1)
        translated_sentence = spanish_tokenizer.decode(translated_tokens)
        return translated_sentence

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')

    # data = [
    #     ["hello", "hola"],
    #     ["how are you", "cómo estás"],
    #     ["good morning", "buenos días"],
    #     ["thank you", "gracias"],
    #     ["see you later", "hasta luego"],
    #     ["good night", "buenas noches"],
    #     ["please", "por favor"],
    #     ["yes", "sí"],
    #     ["no", "no"],
    #     ["what is your name", "cuál es tu nombre"],
    #     ["my name is", "mi nombre es"],
    # ]

    df = pd.read_csv('data/data.csv')
    data = []

    for i in range(len(df)):
        row = [df.iloc[i, 0], df.iloc[i, 1]]
        data.append(row)

    english_tokenizer = SimpleTokenizer()
    spanish_tokenizer = SimpleTokenizer()

    english_sentences = [df.iloc[i, 0] for i in range(len(df))]
    spanish_sentences = [df.iloc[i, 1] for i in range(len(df))]

    english_tokenizer.build_vocab(english_sentences)
    spanish_tokenizer.build_vocab(spanish_sentences)

    max_len = get_max_len(data, english_tokenizer, spanish_tokenizer)
    num_heads = 8
    num_layers = 2
    batch_size = 256
    embed_dim = 512
    hidden_dim = 2048

    english_tensor, spanish_tensor = preprocess_data(data, max_len, english_tokenizer, spanish_tokenizer, device)

    # print(f'English Tensor Shape: {english_tensor.shape}, Spanish Tensor Shape: {spanish_tensor.shape}')
    dataset = TranslationDataset(english_tensor, spanish_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    model = Transformer(
        src_vocab_size=english_tokenizer.vocab_size,
        tgt_vocab_size=spanish_tokenizer.vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        max_len=max_len
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model = train(model, dataloader, criterion, optimizer, device, num_epochs=10)

    input_sentence = 'good start. keep going!'
    translated_sentence = test(model, input_sentence, english_tokenizer, spanish_tokenizer, device, max_length)

    print(f'\n{input_sentence}: {translated_sentence}')
