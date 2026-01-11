import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import random
import re
import json

# =============================================================================
# VOCABULARY (Extended with Q&A tokens)
# =============================================================================

CHARS = (
    'abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    'äöüßÄÖÜ'
    '0123456789'
    ' .,!?;:\'"()-_'
    '\n'
)

SPECIAL_TOKENS = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<Q>', '<A>', '<SEP>']
char_to_idx = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
for c in CHARS:
    if c not in char_to_idx:
        char_to_idx[c] = len(char_to_idx)
idx_to_char = {v: k for k, v in char_to_idx.items()}
VOCAB_SIZE = len(char_to_idx)

# Special token indices
Q_TOKEN = char_to_idx['<Q>']
A_TOKEN = char_to_idx['<A>']
SEP_TOKEN = char_to_idx['<SEP>']
BOS_TOKEN = char_to_idx['<BOS>']
EOS_TOKEN = char_to_idx['<EOS>']
PAD_TOKEN = char_to_idx['<PAD>']

def text_to_indices(text, max_len=128):
    indices = [BOS_TOKEN]
    for c in text[:max_len-2]:
        indices.append(char_to_idx.get(c, char_to_idx['<UNK>']))
    indices.append(EOS_TOKEN)
    while len(indices) < max_len:
        indices.append(PAD_TOKEN)
    return indices[:max_len]

def answer_to_indices(answer, max_len=128):
    indices = [A_TOKEN]
    for c in answer[:max_len-2]:
        indices.append(char_to_idx.get(c, char_to_idx['<UNK>']))
    indices.append(EOS_TOKEN)
    while len(indices) < max_len:
        indices.append(PAD_TOKEN)
    return indices[:max_len]

def question_to_indices(question, max_len=128):
    indices = [Q_TOKEN]
    for c in question[:max_len-2]:
        indices.append(char_to_idx.get(c, char_to_idx['<UNK>']))
    indices.append(EOS_TOKEN)
    while len(indices) < max_len:
        indices.append(PAD_TOKEN)
    return indices[:max_len]

def indices_to_text(indices):
    chars = []
    for idx in indices:
        if idx == PAD_TOKEN or idx == EOS_TOKEN:
            break
        if idx in [BOS_TOKEN, Q_TOKEN, A_TOKEN, SEP_TOKEN]:
            continue
        chars.append(idx_to_char.get(idx, '?'))
    return ''.join(chars)

# =============================================================================
# BRAIN MODULES
# =============================================================================

class DopamineSystem(nn.Module):
    def __init__(self, baseline_tau=0.99):
        super().__init__()
        self.baseline_tau = baseline_tau
        self.register_buffer('reward_baseline', torch.tensor(0.0))
        
    def forward(self, reward):
        with torch.no_grad():
            rpe = reward - self.reward_baseline
            batch_mean = reward.mean() if reward.dim() > 0 else reward
            self.reward_baseline = self.baseline_tau * self.reward_baseline + (1 - self.baseline_tau) * batch_mean
        return rpe

class SerotoninSystem(nn.Module):
    def __init__(self, tau=0.95):
        super().__init__()
        self.tau = tau
        self.register_buffer('serotonin_level', torch.tensor(0.5))
        
    def update(self, reward, confidence):
        with torch.no_grad():
            target = (reward + confidence) / 2
            self.serotonin_level = self.tau * self.serotonin_level + (1 - self.tau) * target
            
    def get_temperature(self, base_temp=0.8):
        return base_temp * max(0.5, min(2.0, 1.5 - self.serotonin_level.item()))

class AcetylcholineSystem(nn.Module):
    """
    Signals expected uncertainty and modulates hierarchy weights.
    Also provides 'precision signaling' to modulate learning rates of different levels.
    """
    def __init__(self, num_levels=6, tau=0.98):
        super().__init__()
        self.tau = tau
        self.num_levels = num_levels
        self.register_buffer('ach_level', torch.tensor(0.5))
        self.register_buffer('weights', torch.ones(num_levels) / num_levels)
        self.register_buffer('loss_variance', torch.tensor(0.0))
        self.register_buffer('precision_weights', torch.ones(num_levels))
        
    def update(self, uncertainty, loss_var=None):
        with torch.no_grad():
            self.ach_level = self.tau * self.ach_level + (1 - self.tau) * uncertainty
            if loss_var is not None:
                self.loss_variance = self.tau * self.loss_variance + (1 - self.tau) * loss_var
            
            # Modulate weights: high ACh/Uncertainty -> focus on lower levels (bottom-up)
            factor = self.ach_level.item()
            new_weights = torch.zeros(self.num_levels, device=self.weights.device)
            for i in range(self.num_levels):
                # Lower i are sensory, higher i are context
                new_weights[i] = (1.0 - i/(self.num_levels-1)) * factor + (i/(self.num_levels-1)) * (1.0 - factor)
            self.weights = F.softmax(new_weights, dim=0)
            
            # Precision signaling: calculate how much each level should learn
            # High uncertainty at top -> lower levels need more learning (high precision)
            for i in range(self.num_levels):
                level_rel = i / (self.num_levels - 1)
                self.precision_weights[i] = 1.0 + (1.0 - level_rel) * self.ach_level - level_rel * (1.0 - self.ach_level)
            self.precision_weights = self.precision_weights.clamp(0.1, 5.0)
            
    def forward(self, embeddings_list):
        return self.weights, self.ach_level, self.precision_weights

class HebbianPlasticity(nn.Module):
    def __init__(self, d_pre, d_post, tau=0.995):
        super().__init__()
        self.tau = tau
        self.register_buffer('association_matrix', torch.zeros(d_pre, d_post))
        
    def update(self, pre, post):
        with torch.no_grad():
            pre = pre.mean(dim=0) if pre.dim() > 1 else pre
            post = post.mean(dim=0) if post.dim() > 1 else post
            hebbian = torch.outer(pre, post)
            self.association_matrix = self.tau * self.association_matrix + (1 - self.tau) * hebbian

class Hippocampus(nn.Module):
    def __init__(self, d_emb, capacity=10000):
        super().__init__()
        self.d_emb = d_emb
        self.capacity = capacity
        self.register_buffer('memory', torch.zeros(capacity, d_emb))
        self.register_buffer('count', torch.tensor(0))
        
    def encode(self, embedding):
        with torch.no_grad():
            if embedding.dim() > 1:
                embedding = embedding.mean(dim=0)
            idx = self.count.item() % self.capacity
            self.memory[idx] = embedding
            self.count += 1

    def retrieve(self, query_emb, k=3):
        """Similarity-based retrieval from memory."""
        with torch.no_grad():
            if self.count == 0:
                return None
            
            # query_emb: [B, d_emb]
            # memory: [capacity, d_emb]
            actual_count = min(self.count.item(), self.capacity)
            mem = self.memory[:actual_count]
            
            # Normalize for cosine similarity
            mem_norm = F.normalize(mem, p=2, dim=1)
            query_norm = F.normalize(query_emb, p=2, dim=1)
            
            # Similarity: [B, actual_count]
            sims = torch.matmul(query_norm, mem_norm.T)
            
            # Top-k
            vals, idxs = sims.topk(min(k, actual_count), dim=1)
            
            # Return mean of top-k retrieved embeddings
            retrieved = mem[idxs].mean(dim=1)
            return retrieved, vals.mean(dim=1)

# =============================================================================
# HIERARCHICAL LAYERS
# =============================================================================

class CharacterEncoder(nn.Module):
    def __init__(self, vocab_size, d_char=128, max_len=128):
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, d_char, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, d_char)
        self.local_cnn = nn.Sequential(
            nn.Conv1d(d_char, d_char, 3, padding=1), nn.GELU(),
            nn.Conv1d(d_char, d_char, 3, padding=1))
        self.norm = nn.LayerNorm(d_char)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.char_embed(x) + self.pos_embed(positions)
        cnn_out = self.local_cnn(x.transpose(1, 2)).transpose(1, 2)
        return self.dropout(self.norm(x + cnn_out))

class SyllableDetector(nn.Module):
    def __init__(self, d_char=128, d_syl=128):
        super().__init__()
        self.lstm = nn.LSTM(d_char, d_char // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.project = nn.Linear(d_char, d_syl)
        self.norm = nn.LayerNorm(d_syl)
        
    def forward(self, char_emb):
        lstm_out, _ = self.lstm(char_emb)
        return self.norm(self.project(lstm_out))

class MorphemeParser(nn.Module):
    def __init__(self, d_syl=128, d_morph=256):
        super().__init__()
        self.lstm = nn.LSTM(d_syl, d_syl, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.project = nn.Sequential(nn.Linear(d_syl * 2, d_morph), nn.LayerNorm(d_morph), nn.GELU())
        
    def forward(self, syl_emb):
        lstm_out, _ = self.lstm(syl_emb)
        return self.project(lstm_out)

class WordComposer(nn.Module):
    def __init__(self, d_morph=256, d_word=256):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_morph, 4, dropout=0.1, batch_first=True)
        self.project = nn.Sequential(nn.Linear(d_morph, d_word), nn.LayerNorm(d_word), nn.GELU())
        
    def forward(self, morph_emb):
        attended, _ = self.attention(morph_emb, morph_emb, morph_emb)
        return self.project(attended)

class PhraseChunker(nn.Module):
    def __init__(self, d_word=256, d_phrase=512):
        super().__init__()
        self.lstm = nn.LSTM(d_word, d_word, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.project = nn.Sequential(nn.Linear(d_word * 2, d_phrase), nn.LayerNorm(d_phrase), nn.GELU())
        
    def forward(self, word_emb):
        lstm_out, _ = self.lstm(word_emb)
        return self.project(lstm_out)

class SentenceEncoder(nn.Module):
    def __init__(self, d_phrase=512, d_sent=512, nhead=8, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_phrase, nhead=nhead, dim_feedforward=d_phrase * 4, dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_phrase) * 0.02)
        self.project = nn.Sequential(nn.Linear(d_phrase, d_sent), nn.LayerNorm(d_sent), nn.GELU())
        
    def forward(self, phrase_emb):
        B = phrase_emb.size(0)
        x = torch.cat([self.cls_token.expand(B, -1, -1), phrase_emb], dim=1)
        x = self.transformer(x)
        return self.project(x[:, 0])

class DiscourseComposer(nn.Module):
    """
    Maintains narrative state across multiple sentences using an attention-based memory buffer.
    Integrates the current sentence embedding with past sentence context.
    """
    def __init__(self, d_sent=512, d_discourse=512, memory_size=10):
        super().__init__()
        self.d_discourse = d_discourse
        self.memory_size = memory_size
        self.attention = nn.MultiheadAttention(d_sent, 8, batch_first=True)
        self.gru = nn.GRUCell(d_sent, d_discourse)
        self.norm = nn.LayerNorm(d_discourse)
        
    def forward(self, sent_emb, hidden_state=None, memory_buffer=None):
        """
        sent_emb: [B, d_sent]
        hidden_state: [B, d_discourse] or None
        memory_buffer: [B, seq_len, d_sent] or None
        """
        B = sent_emb.size(0)
        device = sent_emb.device
        
        if hidden_state is None:
            hidden_state = torch.zeros(B, self.d_discourse, device=device)
        
        if memory_buffer is not None and memory_buffer.size(1) > 0:
            # Attend to past sentences
            attn_out, _ = self.attention(sent_emb.unsqueeze(1), memory_buffer, memory_buffer)
            # Mix current embedding with attended context
            context_emb = (sent_emb + attn_out.squeeze(1)) / 2
        else:
            context_emb = sent_emb
            memory_buffer = torch.zeros(B, 0, sent_emb.size(-1), device=device)
            
        new_hidden = self.gru(context_emb, hidden_state)
        
        # Update memory buffer (rolling window)
        new_memory = torch.cat([memory_buffer, sent_emb.unsqueeze(1)], dim=1)
        if new_memory.size(1) > self.memory_size:
            new_memory = new_memory[:, 1:, :]
            
        return self.norm(new_hidden), new_memory

# =============================================================================
# Q&A AND GENERATION MODULES
# =============================================================================

class QuestionDetector(nn.Module):
    def __init__(self, d_sent=512):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(d_sent, d_sent // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_sent // 2, 2))
    def forward(self, sent_emb):
        return self.classifier(sent_emb)

class QuestionTypeClassifier(nn.Module):
    def __init__(self, d_sent=512, num_types=7):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(d_sent, d_sent // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_sent // 2, num_types))
        self.types = ['was', 'wer', 'wo', 'wann', 'warum', 'wie', 'ja_nein']
    def forward(self, sent_emb):
        return self.classifier(sent_emb)

class ConditionalGenerator(nn.Module):
    def __init__(self, d_sent=512, d_hidden=512, vocab_size=VOCAB_SIZE, max_len=128):
        super().__init__()
        self.max_len, self.vocab_size = max_len, vocab_size
        self.mode_embed = nn.Embedding(2, d_hidden)
        self.sent_project = nn.Linear(d_sent, d_hidden)
        self.char_embed = nn.Embedding(vocab_size, d_hidden)
        self.pos_embed = nn.Embedding(max_len, d_hidden)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_hidden, nhead=8, dim_feedforward=d_hidden * 4, dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.output_head = nn.Linear(d_hidden, vocab_size)
        self.register_buffer('causal_mask', None)
        
    def _get_causal_mask(self, seq_len, device):
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            self.causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self, sent_emb, target_chars, mode=1):
        B = sent_emb.size(0)
        device = sent_emb.device
        if not isinstance(mode, torch.Tensor):
            mode = torch.full((B,), mode, device=device, dtype=torch.long)
        mode_emb = self.mode_embed(mode)
        memory = (self.sent_project(sent_emb) + mode_emb).unsqueeze(1)
        seq_len = target_chars.size(1)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        char_emb = self.char_embed(target_chars) + self.pos_embed(positions)
        output = self.transformer(char_emb, memory, tgt_mask=self._get_causal_mask(seq_len, device))
        return self.output_head(output)

    @torch.no_grad()
    def generate(self, sent_emb, mode=1, max_len=100, temperature=0.8):
        B = sent_emb.size(0)
        device = sent_emb.device
        mode_emb = self.mode_embed(torch.full((B,), mode, device=device, dtype=torch.long))
        memory = (self.sent_project(sent_emb) + mode_emb).unsqueeze(1)
        start_token = Q_TOKEN if mode == 0 else A_TOKEN
        generated = torch.full((B, 1), start_token, dtype=torch.long, device=device)
        for i in range(max_len - 1):
            seq_len = generated.size(1)
            char_emb = self.char_embed(generated) + self.pos_embed(torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1))
            output = self.transformer(char_emb, memory, tgt_mask=self._get_causal_mask(seq_len, device))
            logits = self.output_head(output[:, -1, :]) / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == EOS_TOKEN).all(): break
        return generated

# =============================================================================
# FINAL PHASE 8 MODEL
# =============================================================================

class HierarchicalGermanPhase8(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_char=128, d_syl=128, d_morph=256, d_word=256, d_phrase=512, d_sent=512, max_len=128):
        super().__init__()
        self.vocab_size, self.max_len = vocab_size, max_len
        self.char_encoder = CharacterEncoder(vocab_size, d_char, max_len)
        self.syllable_detector = SyllableDetector(d_char, d_syl)
        self.morpheme_parser = MorphemeParser(d_syl, d_morph)
        self.word_composer = WordComposer(d_morph, d_word)
        self.phrase_chunker = PhraseChunker(d_word, d_phrase)
        self.sentence_encoder = SentenceEncoder(d_phrase, d_sent)
        self.discourse_composer = DiscourseComposer(d_sent, d_sent)
        self.question_detector = QuestionDetector(d_sent)
        self.question_type = QuestionTypeClassifier(d_sent)
        self.generator = ConditionalGenerator(d_sent, d_phrase, vocab_size, max_len)
        self.char_decoder = nn.Sequential(nn.Linear(d_char, d_char * 2), nn.GELU(), nn.Linear(d_char * 2, vocab_size))
        
        # Brain modules
        self.dopamine = DopamineSystem()
        self.serotonin = SerotoninSystem()
        self.acetylcholine = AcetylcholineSystem(num_levels=6)
        self.hebbian = HebbianPlasticity(d_word, d_phrase)
        self.hippocampus = Hippocampus(d_sent, capacity=10000)
        
    def encode(self, char_indices, discourse_state=None, memory_buffer=None):
        char_emb = self.char_encoder(char_indices)
        syl_emb = self.syllable_detector(char_emb)
        morph_emb = self.morpheme_parser(syl_emb)
        word_emb = self.word_composer(morph_emb)
        phrase_emb = self.phrase_chunker(word_emb)
        sent_emb = self.sentence_encoder(phrase_emb)
        
        # 1. Acetylcholine hierarchy modulation
        weights, ach_level, precision = self.acetylcholine([char_emb, syl_emb, morph_emb, word_emb, phrase_emb, sent_emb])
        
        # 2. Hippocampus Retrieval (Knowledge Memory)
        # Use current sentence embedding to query the hippocampus
        retrieval = self.hippocampus.retrieve(sent_emb)
        if retrieval is not None:
            retrieved_mem, mem_sim = retrieval
            # Modulate sent_emb with retrieved knowledge
            # Higher similarity means we trust the memory more
            sent_emb = (sent_emb + retrieved_mem) / 2
        
        # 3. Discourse update with attention memory (Sequential Context)
        new_discourse, new_memory = self.discourse_composer(sent_emb, discourse_state, memory_buffer)
        
        # Modulate sent_emb with discourse context
        sent_emb = (sent_emb + new_discourse) / 2
        
        return sent_emb, word_emb, new_discourse, new_memory, precision
    
    def forward(self, char_indices, target_chars=None, mode=1, discourse_state=None, memory_buffer=None):
        sent_emb, word_emb, new_discourse, new_memory, precision = self.encode(char_indices, discourse_state, memory_buffer)
        is_question = self.question_detector(sent_emb)
        q_type = self.question_type(sent_emb)
        gen_logits = self.generator(sent_emb, target_chars, mode=mode) if target_chars is not None else None
        char_recon = self.char_decoder(self.char_encoder(char_indices))
        if self.training:
            self.hebbian.update(word_emb.mean(dim=1), (sent_emb + new_discourse)/2)
            self.hippocampus.encode(sent_emb)
        return {
            'sent_emb': sent_emb,
            'discourse_state': new_discourse,
            'memory_buffer': new_memory,
            'precision': precision,
            'is_question': is_question,
            'question_type': q_type,
            'gen_logits': gen_logits,
            'char_recon': char_recon,
        }

    @torch.no_grad()
    def ask_question(self, context, discourse_state=None, memory_buffer=None):
        self.eval()
        device = next(self.parameters()).device
        chars = torch.tensor([text_to_indices(context, self.max_len)], device=device)
        sent_emb, _, ds, mb, _ = self.encode(chars, discourse_state, memory_buffer)
        temp = self.serotonin.get_temperature()
        generated = self.generator.generate(sent_emb, mode=0, temperature=temp)
        return indices_to_text(generated[0].cpu().tolist()), ds, mb

    @torch.no_grad()
    def answer_question(self, question, discourse_state=None, memory_buffer=None):
        self.eval()
        device = next(self.parameters()).device
        chars = torch.tensor([text_to_indices(question, self.max_len)], device=device)
        sent_emb, _, ds, mb, _ = self.encode(chars, discourse_state, memory_buffer)
        temp = self.serotonin.get_temperature()
        generated = self.generator.generate(sent_emb, mode=1, temperature=temp)
        return indices_to_text(generated[0].cpu().tolist()), ds, mb

    @torch.no_grad()
    def is_question(self, text, discourse_state=None, memory_buffer=None):
        self.eval()
        device = next(self.parameters()).device
        chars = torch.tensor([text_to_indices(text, self.max_len)], device=device)
        sent_emb, _, _, _, _ = self.encode(chars, discourse_state, memory_buffer)
        return self.question_detector(sent_emb).argmax(dim=-1).item() == 1

    def learn_from_correction(self, text):
        """
        Online learning from user correction.
        Strengthens Hebbian associations and stores in Hippocampus.
        """
        self.train()
        device = next(self.parameters()).device
        chars = torch.tensor([text_to_indices(text, self.max_len)], device=device)
        
        # Encoding triggers Hebbian and Hippocampus updates in forward() if self.training is True
        # But we want to be explicit here
        char_emb = self.char_encoder(chars)
        syl_emb = self.syllable_detector(char_emb)
        morph_emb = self.morpheme_parser(syl_emb)
        word_emb = self.word_composer(morph_emb)
        phrase_emb = self.phrase_chunker(word_emb)
        sent_emb = self.sentence_encoder(phrase_emb)
        
        # Update brain modules
        self.hebbian.update(word_emb.mean(dim=1), sent_emb)
        self.hippocampus.encode(sent_emb)
        
        # Signal success via serotonin (mood improvement)
        self.serotonin.update(torch.tensor(1.0, device=device), torch.tensor(1.0, device=device))
        
        self.eval()
        return "Gelernt!"

# =============================================================================
# DATA AND TRAINING
# =============================================================================

def load_qa_data(file_path="german_qa_dataset.jsonl"):
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            pairs.append((data['question'], data['answer']))
    random.shuffle(pairs)
    return pairs

def load_discourse_data(file_path="german_discourse_dataset.jsonl"):
    """Load multi-sentence narratives."""
    narratives = []
    if os.path.exists(file_path):
        print(f"Loading discourse data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                narratives.append(data['sentences'])
    return narratives

def train(model, qa_pairs, discourse_data, device, epochs=10, batch_size=32):
    val_size = int(len(qa_pairs) * 0.05)
    train_qa, val_qa = qa_pairs[:-val_size], qa_pairs[-val_size:]
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scaler = GradScaler()
    
    from hierarchical_german_phase6_qa import prepare_qa_batch, get_question_type
    
    # Process types for train_qa
    train_triplets = [(q, a, get_question_type(q)) for q, a in train_qa]
    
    # Track loss history for variance calculation
    loss_history = []
    max_history = 100

    for epoch in range(1, epochs + 1):
        random.shuffle(train_triplets)
        random.shuffle(discourse_data)
        
        num_batches = len(train_triplets) // batch_size
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{epochs}")
        
        for b_idx in pbar:
            # 1. QA Batch
            batch = train_triplets[b_idx*batch_size : (b_idx+1)*batch_size]
            if len(batch) < 2: continue
            
            stmt_batch = [a for _, a, _ in batch[:len(batch)//2]]
            q_in, t_chars, q_lab, t_lab, modes = prepare_qa_batch(batch, statements=stmt_batch)
            q_in, t_chars, q_lab, t_lab, modes = q_in.to(device), t_chars.to(device), q_lab.to(device), t_lab.to(device), modes.to(device)
            
            optimizer.zero_grad()
            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                outputs = model(q_in, t_chars[:, :-1], mode=modes)
                
                q_det_loss = F.cross_entropy(outputs['is_question'], q_lab)
                q_mask = (q_lab == 1)
                type_loss = F.cross_entropy(outputs['question_type'][q_mask], t_lab[q_mask]) if q_mask.any() else 0
                gen_logits = outputs['gen_logits']
                targets = t_chars[:, 1:]
                raw_gen_loss = F.cross_entropy(gen_logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1), ignore_index=PAD_TOKEN, reduction='none').reshape(gen_logits.size(0), -1).mean(dim=1)
                
                # ACh and Dopamine updates
                current_loss_val = raw_gen_loss.mean().detach()
                loss_history.append(current_loss_val.item())
                if len(loss_history) > max_history: loss_history.pop(0)
                
                loss_var = torch.tensor(loss_history).var() if len(loss_history) > 1 else torch.tensor(0.0)
                model.acetylcholine.update(current_loss_val, loss_var=loss_var.to(device))
                
                # Get precision weights from ACh
                _, _, precision = model.acetylcholine([]) # Dummy list as forward doesn't use it yet
                
                rpe = model.dopamine(-raw_gen_loss)
                # Modulate gen_loss with Dopamine (RPE) and ACh (precision for top level)
                gen_loss = (raw_gen_loss * torch.exp(rpe).clamp(0.5, 2.0) * precision[-1]).mean()
                
                # Modulate losses based on precision signaling
                # Top-level tasks (QA, Type) use higher-level precision
                qa_loss = gen_loss + q_det_loss * 0.5 * precision[-1] + type_loss * 0.3 * precision[-1]
                
            scaler.scale(qa_loss).backward()
            
            # 2. Discourse Batch (Context persistence training)
            disc_batch = discourse_data[b_idx % len(discourse_data)]
            discourse_state = None
            memory_buffer = None
            total_disc_loss = 0
            
            for sentence in disc_batch:
                s_indices = torch.tensor([text_to_indices(sentence)], device=device)
                with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    outputs = model(s_indices, discourse_state=discourse_state, memory_buffer=memory_buffer)
                    discourse_state = outputs['discourse_state']
                    memory_buffer = outputs['memory_buffer']
                    recon_loss = F.cross_entropy(outputs['char_recon'].view(-1, model.vocab_size), s_indices.view(-1), ignore_index=PAD_TOKEN)
                    total_disc_loss += recon_loss * 0.1
            
            scaler.scale(total_disc_loss).backward()
            
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix({'qa': f'{qa_loss.item():.3f}', 'disc': f'{total_disc_loss.item():.3f}', 'ach': f'{model.acetylcholine.ach_level.item():.2f}'})

        # Save checkpoint
        torch.save(model.state_dict(), "checkpoints/phase8_discourse_best.pth")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs("checkpoints", exist_ok=True)
    model = HierarchicalGermanPhase8().to(device)
    
    # Load previous best
    checkpoint_path = "checkpoints/phase7_ach_best.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading Phase 7 weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model_dict = model.state_dict()
        
        # Filter out mismatching keys (like hippocampus.memory if size changed)
        pretrained = {k: v for k, v in state_dict.items() 
                     if k in model_dict and model_dict[k].shape == v.shape}
        print(f"  Loading {len(pretrained)}/{len(state_dict)} parameters")
        model_dict.update(pretrained)
        model.load_state_dict(model_dict, strict=False)
    
    qa_pairs = load_qa_data("german_qa_dataset.jsonl")
    disc_data = load_discourse_data("german_discourse_dataset.jsonl")
    
    print(f"Loaded {len(qa_pairs)} Q&A pairs and {len(disc_data)} discourse narratives.")
    train(model, qa_pairs, disc_data, device, epochs=5, batch_size=64)

if __name__ == "__main__":
    main()
