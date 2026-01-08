#!/usr/bin/env python3
"""
Hierarchical German Language Model - Phase 6: Question & Answer

New capabilities:
1. Question detection - identify if input is a question
2. Question generation - generate questions about a topic
3. Answer generation - generate answers to questions

Uses Q&A pairs for training with special tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from datasets import load_dataset
from tqdm import tqdm
import os
import random
import re

# =============================================================================
# VOCABULARY (Extended with Q&A tokens)
# =============================================================================

CHARS = (
    'abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    'äöüßÄÖÜ'
    '0123456789'
    ' .,!?;:\'"()-\n'
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
    """Encode answer only: <A>answer<EOS>"""
    indices = [A_TOKEN]
    for c in answer[:max_len-2]:
        indices.append(char_to_idx.get(c, char_to_idx['<UNK>']))
    indices.append(EOS_TOKEN)
    while len(indices) < max_len:
        indices.append(PAD_TOKEN)
    return indices[:max_len]

def question_to_indices(question, max_len=128):
    """Encode question: <Q>question<EOS>"""
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
# BRAIN MODULES (from Phase 5)
# =============================================================================

class DopamineSystem(nn.Module):
    def __init__(self, baseline_tau=0.99):
        super().__init__()
        self.baseline_tau = baseline_tau
        self.register_buffer('reward_baseline', torch.tensor(0.0))
        
    def forward(self, reward):
        with torch.no_grad():
            rpe = reward - self.reward_baseline
            self.reward_baseline = self.baseline_tau * self.reward_baseline + (1 - self.baseline_tau) * reward
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

class HebbianPlasticity(nn.Module):
    def __init__(self, d_pre, d_post, tau=0.995):
        super().__init__()
        self.tau = tau
        self.register_buffer('association_matrix', torch.zeros(d_pre, d_post))
        
    def update(self, pre, post):
        with torch.no_grad():
            pre = pre.mean(dim=0) if pre.dim() > 1 else pre
            post = post.mean(dim=0) if post.dim() > 1 else post
            if pre.dim() > 1: pre = pre.mean(dim=0)
            if post.dim() > 1: post = post.mean(dim=0)
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
        self.lstm = nn.LSTM(d_char, d_char // 2, num_layers=2, 
                           batch_first=True, bidirectional=True, dropout=0.1)
        self.project = nn.Linear(d_char, d_syl)
        self.norm = nn.LayerNorm(d_syl)
        
    def forward(self, char_emb):
        lstm_out, _ = self.lstm(char_emb)
        return self.norm(self.project(lstm_out))

class MorphemeParser(nn.Module):
    def __init__(self, d_syl=128, d_morph=256):
        super().__init__()
        self.lstm = nn.LSTM(d_syl, d_syl, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=0.1)
        self.project = nn.Sequential(
            nn.Linear(d_syl * 2, d_morph), nn.LayerNorm(d_morph), nn.GELU())
        
    def forward(self, syl_emb):
        lstm_out, _ = self.lstm(syl_emb)
        return self.project(lstm_out)

class WordComposer(nn.Module):
    def __init__(self, d_morph=256, d_word=256):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_morph, 4, dropout=0.1, batch_first=True)
        self.project = nn.Sequential(
            nn.Linear(d_morph, d_word), nn.LayerNorm(d_word), nn.GELU())
        
    def forward(self, morph_emb):
        attended, _ = self.attention(morph_emb, morph_emb, morph_emb)
        return self.project(attended)

class PhraseChunker(nn.Module):
    def __init__(self, d_word=256, d_phrase=512):
        super().__init__()
        self.lstm = nn.LSTM(d_word, d_word, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=0.1)
        self.project = nn.Sequential(
            nn.Linear(d_word * 2, d_phrase), nn.LayerNorm(d_phrase), nn.GELU())
        
    def forward(self, word_emb):
        lstm_out, _ = self.lstm(word_emb)
        return self.project(lstm_out)

class SentenceEncoder(nn.Module):
    def __init__(self, d_phrase=512, d_sent=512, nhead=8, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_phrase, nhead=nhead, dim_feedforward=d_phrase * 4,
            dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_phrase) * 0.02)
        self.project = nn.Sequential(nn.Linear(d_phrase, d_sent), nn.LayerNorm(d_sent), nn.GELU())
        
    def forward(self, phrase_emb):
        B = phrase_emb.size(0)
        x = torch.cat([self.cls_token.expand(B, -1, -1), phrase_emb], dim=1)
        x = self.transformer(x)
        return self.project(x[:, 0])

# =============================================================================
# Q&A SPECIFIC MODULES
# =============================================================================

class QuestionDetector(nn.Module):
    """Detect if input is a question."""
    def __init__(self, d_sent=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_sent, d_sent // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_sent // 2, 2)  # [not_question, is_question]
        )
        
    def forward(self, sent_emb):
        return self.classifier(sent_emb)

class QuestionTypeClassifier(nn.Module):
    """Classify question type: was/wer/wo/wann/warum/wie/welche."""
    def __init__(self, d_sent=512, num_types=7):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_sent, d_sent // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_sent // 2, num_types)
        )
        # Question types
        self.types = ['was', 'wer', 'wo', 'wann', 'warum', 'wie', 'welche']
        
    def forward(self, sent_emb):
        return self.classifier(sent_emb)

class ConditionalGenerator(nn.Module):
    """Generate text conditioned on sentence embedding and mode (Q or A)."""
    def __init__(self, d_sent=512, d_hidden=512, vocab_size=VOCAB_SIZE, max_len=128):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        
        # Mode embedding (question vs answer generation)
        self.mode_embed = nn.Embedding(2, d_hidden)  # 0=question, 1=answer
        
        # Conditioning
        self.sent_project = nn.Linear(d_sent, d_hidden)
        self.char_embed = nn.Embedding(vocab_size, d_hidden)
        self.pos_embed = nn.Embedding(max_len, d_hidden)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_hidden, nhead=8, dim_feedforward=d_hidden * 4,
            dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.output_head = nn.Linear(d_hidden, vocab_size)
        self.register_buffer('causal_mask', None)
        
    def _get_causal_mask(self, seq_len, device):
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            self.causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self, sent_emb, target_chars, mode=1):
        """mode: 0=generate question, 1=generate answer"""
        B = sent_emb.size(0)
        device = sent_emb.device
        
        # Add mode embedding to memory
        mode_emb = self.mode_embed(torch.full((B,), mode, device=device, dtype=torch.long))
        memory = (self.sent_project(sent_emb) + mode_emb).unsqueeze(1)
        
        seq_len = target_chars.size(1)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        char_emb = self.char_embed(target_chars) + self.pos_embed(positions)
        
        causal_mask = self._get_causal_mask(seq_len, device)
        output = self.transformer(char_emb, memory, tgt_mask=causal_mask)
        return self.output_head(output)
    
    @torch.no_grad()
    def generate(self, sent_emb, mode=1, max_len=100, temperature=0.8, top_k=50, top_p=0.9):
        """Generate question (mode=0) or answer (mode=1)."""
        B = sent_emb.size(0)
        device = sent_emb.device
        
        mode_emb = self.mode_embed(torch.full((B,), mode, device=device, dtype=torch.long))
        memory = (self.sent_project(sent_emb) + mode_emb).unsqueeze(1)
        
        # Start token depends on mode
        start_token = Q_TOKEN if mode == 0 else A_TOKEN
        generated = torch.full((B, 1), start_token, dtype=torch.long, device=device)
        
        for i in range(max_len - 1):
            seq_len = generated.size(1)
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
            char_emb = self.char_embed(generated) + self.pos_embed(positions)
            causal_mask = self._get_causal_mask(seq_len, device)
            
            output = self.transformer(char_emb, memory, tgt_mask=causal_mask)
            logits = self.output_head(output[:, -1, :]) / temperature
            
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token == EOS_TOKEN).all() or (next_token == SEP_TOKEN).all():
                break
        
        return generated

# =============================================================================
# COMPLETE Q&A MODEL
# =============================================================================

class HierarchicalGermanQA(nn.Module):
    """Full model with Q&A capabilities."""
    def __init__(self, vocab_size=VOCAB_SIZE, d_char=128, d_syl=128, 
                 d_morph=256, d_word=256, d_phrase=512, d_sent=512, max_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # Encoders
        self.char_encoder = CharacterEncoder(vocab_size, d_char, max_len)
        self.syllable_detector = SyllableDetector(d_char, d_syl)
        self.morpheme_parser = MorphemeParser(d_syl, d_morph)
        self.word_composer = WordComposer(d_morph, d_word)
        self.phrase_chunker = PhraseChunker(d_word, d_phrase)
        self.sentence_encoder = SentenceEncoder(d_phrase, d_sent)
        
        # Q&A heads
        self.question_detector = QuestionDetector(d_sent)
        self.question_type = QuestionTypeClassifier(d_sent)
        self.generator = ConditionalGenerator(d_sent, d_phrase, vocab_size, max_len)
        
        # Reconstruction decoder
        self.char_decoder = nn.Sequential(
            nn.Linear(d_char, d_char * 2), nn.GELU(),
            nn.Linear(d_char * 2, vocab_size))
        
        # Brain modules
        self.dopamine = DopamineSystem()
        self.serotonin = SerotoninSystem()
        self.hebbian = HebbianPlasticity(d_word, d_phrase)
        self.hippocampus = Hippocampus(d_word, capacity=10000)
        
    def encode(self, char_indices):
        """Encode text to sentence embedding."""
        char_emb = self.char_encoder(char_indices)
        syl_emb = self.syllable_detector(char_emb)
        morph_emb = self.morpheme_parser(syl_emb)
        word_emb = self.word_composer(morph_emb)
        phrase_emb = self.phrase_chunker(word_emb)
        sent_emb = self.sentence_encoder(phrase_emb)
        return sent_emb, word_emb
    
    def forward(self, char_indices, target_chars=None, mode=1):
        """Forward pass with optional generation target."""
        char_emb = self.char_encoder(char_indices)
        syl_emb = self.syllable_detector(char_emb)
        morph_emb = self.morpheme_parser(syl_emb)
        word_emb = self.word_composer(morph_emb)
        phrase_emb = self.phrase_chunker(word_emb)
        sent_emb = self.sentence_encoder(phrase_emb)
        
        # Q&A predictions
        is_question = self.question_detector(sent_emb)
        q_type = self.question_type(sent_emb)
        
        # Generation
        if target_chars is not None:
            gen_logits = self.generator(sent_emb, target_chars, mode=mode)
        else:
            gen_logits = None
        
        # Reconstruction
        char_recon = self.char_decoder(char_emb)
        
        # Brain updates
        if self.training:
            self.hebbian.update(word_emb.mean(dim=1), phrase_emb.mean(dim=1))
            self.hippocampus.encode(word_emb.mean(dim=1))
        
        return {
            'sent_emb': sent_emb,
            'is_question': is_question,
            'question_type': q_type,
            'gen_logits': gen_logits,
            'char_recon': char_recon,
        }
    
    @torch.no_grad()
    def ask_question(self, context, temperature=0.8):
        """Generate a question about the given context."""
        self.eval()
        device = next(self.parameters()).device
        chars = torch.tensor([text_to_indices(context, self.max_len)], device=device)
        sent_emb, _ = self.encode(chars)
        
        temp = self.serotonin.get_temperature(temperature)
        generated = self.generator.generate(sent_emb, mode=0, temperature=temp)
        return indices_to_text(generated[0].cpu().tolist())
    
    @torch.no_grad()
    def answer_question(self, question, temperature=0.8):
        """Generate an answer to the given question."""
        self.eval()
        device = next(self.parameters()).device
        chars = torch.tensor([text_to_indices(question, self.max_len)], device=device)
        sent_emb, _ = self.encode(chars)
        
        temp = self.serotonin.get_temperature(temperature)
        generated = self.generator.generate(sent_emb, mode=1, temperature=temp)
        return indices_to_text(generated[0].cpu().tolist())
    
    @torch.no_grad()
    def is_question(self, text):
        """Check if text is a question."""
        self.eval()
        device = next(self.parameters()).device
        chars = torch.tensor([text_to_indices(text, self.max_len)], device=device)
        sent_emb, _ = self.encode(chars)
        logits = self.question_detector(sent_emb)
        return logits.argmax(dim=-1).item() == 1
    
    def load_phase5_weights(self, checkpoint_path):
        print(f"Loading Phase 5 weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model_dict = self.state_dict()
        pretrained = {k: v for k, v in state_dict.items() 
                     if k in model_dict and model_dict[k].shape == v.shape}
        print(f"  Loading {len(pretrained)}/{len(state_dict)} parameters")
        model_dict.update(pretrained)
        self.load_state_dict(model_dict, strict=False)

# =============================================================================
# Q&A DATASET
# =============================================================================

GERMAN_QA_PAIRS = [
    # Simple factual questions
    ("Wie heißt du?", "Ich bin ein Sprachmodell."),
    ("Was ist das?", "Das ist ein Buch."),
    ("Wo ist der Hund?", "Der Hund ist im Garten."),
    ("Wer ist das?", "Das ist meine Mutter."),
    ("Wann kommst du?", "Ich komme morgen."),
    ("Warum bist du traurig?", "Ich bin müde."),
    ("Wie geht es dir?", "Mir geht es gut, danke."),
    
    # Object questions
    ("Was ist ein Apfel?", "Ein Apfel ist eine Frucht."),
    ("Was ist ein Hund?", "Ein Hund ist ein Tier."),
    ("Was ist eine Katze?", "Eine Katze ist ein Haustier."),
    ("Was ist ein Baum?", "Ein Baum ist eine Pflanze."),
    ("Was ist die Sonne?", "Die Sonne ist ein Stern."),
    
    # Location questions
    ("Wo wohnst du?", "Ich wohne in Berlin."),
    ("Wo ist die Schule?", "Die Schule ist in der Stadt."),
    ("Wo ist das Buch?", "Das Buch liegt auf dem Tisch."),
    ("Wo sind die Kinder?", "Die Kinder spielen im Park."),
    
    # Time questions
    ("Wann ist Weihnachten?", "Weihnachten ist im Dezember."),
    ("Wann gehst du schlafen?", "Ich gehe um zehn Uhr schlafen."),
    ("Wann beginnt die Schule?", "Die Schule beginnt um acht Uhr."),
    
    # How questions
    ("Wie alt bist du?", "Ich bin zehn Jahre alt."),
    ("Wie viel kostet das?", "Das kostet fünf Euro."),
    ("Wie schmeckt der Kuchen?", "Der Kuchen schmeckt sehr gut."),
    ("Wie ist das Wetter?", "Das Wetter ist heute schön."),
    
    # Why questions
    ("Warum ist der Himmel blau?", "Das Licht wird gestreut."),
    ("Warum regnet es?", "Wolken bringen den Regen."),
    ("Warum lernst du Deutsch?", "Deutsch ist eine schöne Sprache."),
    
    # Yes/No questions
    ("Hast du Hunger?", "Ja, ich habe Hunger."),
    ("Ist das dein Buch?", "Nein, das ist nicht mein Buch."),
    ("Kannst du schwimmen?", "Ja, ich kann schwimmen."),
    ("Magst du Schokolade?", "Ja, ich mag Schokolade sehr."),
    
    # Story-based Q&A
    ("Was macht die Katze?", "Die Katze schläft auf dem Sofa."),
    ("Was isst der Junge?", "Der Junge isst einen Apfel."),
    ("Wohin geht das Mädchen?", "Das Mädchen geht zur Schule."),
    ("Wer hat den Ball?", "Der Hund hat den Ball."),
]

def generate_qa_variations():
    """Generate more Q&A pairs with variations."""
    templates = [
        # What is X?
        [("Was ist ein {noun}?", "Ein {noun} ist {desc}."),
         [("Haus", "ein Gebäude"), ("Auto", "ein Fahrzeug"), ("Vogel", "ein Tier"),
          ("Blume", "eine Pflanze"), ("Berg", "sehr hoch"), ("Fluss", "ein Gewässer")]],
        
        # Where is X?
        [("Wo ist der {noun}?", "Der {noun} ist {loc}."),
         [("Vater", "bei der Arbeit"), ("Lehrer", "in der Schule"), ("Arzt", "im Krankenhaus"),
          ("Ball", "unter dem Bett"), ("Schlüssel", "auf dem Tisch")]],
        
        # Who is X?
        [("Wer ist {name}?", "{name} ist {role}."),
         [("Anna", "meine Schwester"), ("Peter", "mein Freund"), ("Frau Müller", "die Lehrerin"),
          ("Herr Schmidt", "der Nachbar"), ("Oma", "sehr lieb")]],
    ]
    
    pairs = []
    for template_pair, variations in templates:
        q_template, a_template = template_pair
        for var in variations:
            if len(var) == 2:
                q = q_template.format(noun=var[0], name=var[0])
                a = a_template.format(noun=var[0], name=var[0], desc=var[1], loc=var[1], role=var[1])
                pairs.append((q, a))
    return pairs

def load_qa_data():
    """Load Q&A training data."""
    pairs = GERMAN_QA_PAIRS.copy()
    pairs.extend(generate_qa_variations())
    random.shuffle(pairs)
    return pairs

# =============================================================================
# TRAINING
# =============================================================================

def prepare_qa_batch(qa_pairs, max_len=64):
    """Prepare batch of Q&A pairs - question as input, answer as target."""
    questions = []
    answers = []
    q_labels = []
    
    for q, a in qa_pairs:
        # Question as input (for encoding)
        q_seq = question_to_indices(q, max_len)
        
        # Answer as target (for generation)
        a_seq = answer_to_indices(a, max_len)
        
        questions.append(q_seq)
        answers.append(a_seq)
        q_labels.append(1)  # All are questions
    
    return (torch.tensor(questions), 
            torch.tensor(answers), 
            torch.tensor(q_labels))

def test_qa(model, device):
    """Test Q&A capabilities."""
    model.eval()
    print("\n" + "="*60)
    print("Q&A TEST")
    print("="*60)
    
    # Test question detection
    test_sentences = [
        ("Wie heißt du?", True),
        ("Der Hund schläft.", False),
        ("Wo ist die Katze?", True),
        ("Ich gehe nach Hause.", False),
        ("Warum regnet es?", True),
    ]
    
    print("\n📋 Question Detection:")
    correct = 0
    for text, expected in test_sentences:
        is_q = model.is_question(text)
        status = "✅" if is_q == expected else "❌"
        correct += 1 if is_q == expected else 0
        print(f"  {status} '{text}' → {'Question' if is_q else 'Statement'}")
    print(f"  Accuracy: {correct}/{len(test_sentences)}")
    
    # Test answer generation
    print("\n💬 Answer Generation:")
    questions = [
        "Wie heißt du?",
        "Was ist ein Hund?",
        "Wo ist die Katze?",
        "Wie ist das Wetter?",
        "Warum lernst du Deutsch?",
    ]
    for q in questions:
        answer = model.answer_question(q)
        print(f"  Q: {q}")
        print(f"  A: {answer}")
        print()
    
    # Test question generation
    print("❓ Question Generation:")
    contexts = [
        "Der Hund spielt im Garten.",
        "Das Wetter ist heute schön.",
        "Die Kinder essen Kuchen.",
    ]
    for ctx in contexts:
        question = model.ask_question(ctx)
        print(f"  Context: {ctx}")
        print(f"  Question: {question}")
        print()
    
    print("="*60)
    model.train()

def train(model, qa_pairs, device, epochs=15, batch_size=16, lr=2e-4):
    """Train Q&A model."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(qa_pairs) // batch_size + 1)
    scaler = GradScaler()
    
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        random.shuffle(qa_pairs)
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(range(0, len(qa_pairs), batch_size), desc=f"Epoch {epoch}/{epochs}")
        for i in pbar:
            batch = qa_pairs[i:i+batch_size]
            if len(batch) < 2:
                continue
            
            q_chars, a_chars, q_labels = prepare_qa_batch(batch)
            q_chars = q_chars.to(device)
            a_chars = a_chars.to(device)
            q_labels = q_labels.to(device)
            
            optimizer.zero_grad()
            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                # Forward: encode question, generate answer
                outputs = model(q_chars, a_chars[:, :-1], mode=1)
                
                # Question detection loss
                q_det_loss = F.cross_entropy(outputs['is_question'], q_labels)
                
                # Answer generation loss (predict answer from question encoding)
                gen_logits = outputs['gen_logits']
                targets = a_chars[:, 1:]  # Answer tokens shifted
                gen_loss = F.cross_entropy(
                    gen_logits.reshape(-1, model.vocab_size),
                    targets.reshape(-1),
                    ignore_index=PAD_TOKEN,
                    label_smoothing=0.1
                )
                
                # Reconstruction loss
                recon_loss = F.cross_entropy(
                    outputs['char_recon'].view(-1, model.vocab_size),
                    q_chars.view(-1),
                    ignore_index=PAD_TOKEN
                ) * 0.1
                
                loss = gen_loss + q_det_loss * 0.5 + recon_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 5 == 0:
                pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"\nEpoch {epoch}: Loss={avg_loss:.4f}")
        
        test_qa(model, device)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/phase6_qa_best.pth")
            print(f"💾 Saved (loss={avg_loss:.4f})")
    
    print("\n✅ Training complete!")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs("checkpoints", exist_ok=True)
    
    model = HierarchicalGermanQA().to(device)
    
    # Try to load Phase 5 weights
    if os.path.exists("checkpoints/phase5_best.pth"):
        model.load_phase5_weights("checkpoints/phase5_best.pth")
    
    # Load Q&A data
    qa_pairs = load_qa_data()
    print(f"Loaded {len(qa_pairs)} Q&A pairs")
    
    # Augment with repeated pairs for more training data
    qa_pairs = qa_pairs * 50  # Repeat dataset more
    
    print("\n🎓 Training Q&A capabilities...")
    train(model, qa_pairs, device, epochs=25, batch_size=16)
    
    # Final demo
    print("\n" + "="*60)
    print("FINAL Q&A DEMO")
    print("="*60)
    model.load_state_dict(torch.load("checkpoints/phase6_qa_best.pth", map_location=device))
    test_qa(model, device)

if __name__ == "__main__":
    main()
