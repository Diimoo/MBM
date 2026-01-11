#!/usr/bin/env python3
"""
Enhanced Q&A training with:
- Context retrieval for factual answers
- Extended epochs with learning rate decay
- Combined corpus (21k+ pairs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from hierarchical_german_phase6_qa import (
    HierarchicalGermanQA, char_to_idx, idx_to_char, VOCAB_SIZE,
    PAD_TOKEN, Q_TOKEN, A_TOKEN, EOS_TOKEN,
    text_to_indices, answer_to_indices, indices_to_text
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CORPUS_FILE = Path("combined_german_qa.jsonl")
CHECKPOINT_PATH = Path("checkpoints/phase6_qa_best.pth")
OUTPUT_CHECKPOINT = Path("checkpoints/phase6_qa_retrieval.pth")


class ContextRetriever:
    """Simple TF-IDF based context retriever for factual answers."""
    
    def __init__(self):
        self.documents = []  # List of (context, answer) pairs
        self.vocab = defaultdict(int)
        self.doc_freqs = defaultdict(int)
        self.tfidf_cache = {}
    
    def add_document(self, context, answer):
        """Add a document to the retrieval index."""
        if context and len(context) > 20:
            self.documents.append((context, answer))
            words = set(context.lower().split())
            for word in words:
                self.doc_freqs[word] += 1
                self.vocab[word] += 1
    
    def build_index(self):
        """Build TF-IDF index."""
        print(f"   Building retrieval index with {len(self.documents)} documents...")
        self.n_docs = len(self.documents)
        # Pre-compute IDF
        self.idf = {}
        for word, df in self.doc_freqs.items():
            self.idf[word] = np.log(self.n_docs / (df + 1))
    
    def get_tfidf(self, text):
        """Compute TF-IDF vector for text."""
        words = text.lower().split()
        tf = defaultdict(int)
        for w in words:
            tf[w] += 1
        
        vector = {}
        for w, count in tf.items():
            if w in self.idf:
                vector[w] = (count / len(words)) * self.idf[w]
        return vector
    
    def cosine_sim(self, v1, v2):
        """Compute cosine similarity between two sparse vectors."""
        common = set(v1.keys()) & set(v2.keys())
        if not common:
            return 0.0
        
        dot = sum(v1[w] * v2[w] for w in common)
        norm1 = np.sqrt(sum(v ** 2 for v in v1.values()))
        norm2 = np.sqrt(sum(v ** 2 for v in v2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
    
    def retrieve(self, query, top_k=3):
        """Retrieve top-k relevant contexts for a query."""
        if not self.documents:
            return []
        
        query_vec = self.get_tfidf(query)
        
        scores = []
        for i, (context, answer) in enumerate(self.documents):
            # Cache document vectors
            if i not in self.tfidf_cache:
                self.tfidf_cache[i] = self.get_tfidf(context)
            
            score = self.cosine_sim(query_vec, self.tfidf_cache[i])
            scores.append((score, context, answer))
        
        # Sort by score
        scores.sort(reverse=True, key=lambda x: x[0])
        return scores[:top_k]


class EnhancedQADataset(Dataset):
    """Dataset with context retrieval support."""
    
    def __init__(self, qa_pairs, retriever=None, max_len=128):
        self.qa_pairs = qa_pairs
        self.retriever = retriever
        self.max_len = max_len
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        pair = self.qa_pairs[idx]
        question = pair["question"]
        answer = pair["answer"]
        context = pair.get("context", "")
        
        # If no context but retriever available, try to retrieve
        if not context and self.retriever and random.random() < 0.3:
            retrieved = self.retriever.retrieve(question, top_k=1)
            if retrieved and retrieved[0][0] > 0.1:
                context = retrieved[0][1][:200]
        
        # Encode question with optional context
        if context:
            q_text = f"{context[:100]} {question}"
        else:
            q_text = question
        
        q_indices = text_to_indices(q_text, self.max_len)
        a_indices = answer_to_indices(answer, self.max_len)
        
        return {
            "q_chars": torch.tensor(q_indices, dtype=torch.long),
            "a_chars": torch.tensor(a_indices, dtype=torch.long),
            "question": question,
            "answer": answer
        }


def load_corpus():
    """Load combined Q&A corpus."""
    pairs = []
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line.strip()))
    return pairs


def test_qa_with_retrieval(model, retriever):
    """Test Q&A with retrieval augmentation."""
    model.eval()
    
    print("\n" + "=" * 60)
    print("Q&A TEST (With Retrieval)")
    print("=" * 60)
    
    test_questions = [
        "Was ist die Hauptstadt von Deutschland?",
        "Wann fiel die Berliner Mauer?",
        "Wer schrieb Faust?",
        "Wie viele Bundesländer hat Deutschland?",
        "Was ist der höchste Berg Deutschlands?",
    ]
    
    print("\n💬 Answer Generation:")
    for q in test_questions:
        # Try retrieval first
        retrieved = retriever.retrieve(q, top_k=1) if retriever else []
        context_hint = ""
        if retrieved and retrieved[0][0] > 0.15:
            context_hint = f" [Context: {retrieved[0][1][:50]}...]"
        
        try:
            answer = model.answer_question(q, temperature=0.7)
            print(f"  Q: {q}{context_hint}")
            print(f"  A: {answer}\n")
        except Exception as e:
            print(f"  Q: {q}")
            print(f"  A: [Error: {e}]\n")


def train():
    """Train with extended epochs and LR decay."""
    print("=" * 60)
    print("🇩🇪 Enhanced Q&A Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # Load corpus
    print("\n📥 Loading combined corpus...")
    pairs = load_corpus()
    print(f"   Total pairs: {len(pairs)}")
    
    # Build retriever from contexts
    print("\n🔍 Building context retriever...")
    retriever = ContextRetriever()
    for p in pairs:
        if p.get("context"):
            retriever.add_document(p["context"], p["answer"])
    retriever.build_index()
    
    # Split data
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.95)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    print(f"   Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    # Create datasets
    train_dataset = EnhancedQADataset(train_pairs, retriever)
    val_dataset = EnhancedQADataset(val_pairs, retriever)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Load model
    print("\n📦 Loading model...")
    model = HierarchicalGermanQA(VOCAB_SIZE).to(DEVICE)
    if CHECKPOINT_PATH.exists():
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        state_dict = {k: v for k, v in state_dict.items() if 'causal_mask' not in k}
        model.load_state_dict(state_dict, strict=False)
        print(f"   ✅ Loaded from {CHECKPOINT_PATH}")
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    
    # Learning rate scheduler with warmup and cosine decay
    num_epochs = 20
    warmup_epochs = 2
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    
    # Training loop
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    
    print(f"\n🚀 Training for {num_epochs} epochs with LR decay...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            q_chars = batch["q_chars"].to(DEVICE)
            a_chars = batch["a_chars"].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(q_chars, target_chars=a_chars[:, :-1], mode=1)
            
            gen_logits = outputs["gen_logits"]
            targets = a_chars[:, 1:]
            loss = criterion(gen_logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                q_chars = batch["q_chars"].to(DEVICE)
                a_chars = batch["a_chars"].to(DEVICE)
                
                outputs = model(q_chars, target_chars=a_chars[:, :-1], mode=1)
                gen_logits = outputs["gen_logits"]
                targets = a_chars[:, 1:]
                loss = criterion(gen_logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, LR={scheduler.get_last_lr()[0]:.2e}")
        
        # Test every 5 epochs
        if (epoch + 1) % 5 == 0:
            test_qa_with_retrieval(model, retriever)
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_CHECKPOINT)
            print(f"💾 Saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                break
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Checkpoint: {OUTPUT_CHECKPOINT}")
    
    # Final test
    test_qa_with_retrieval(model, retriever)
    
    # Save retriever for chat
    print("\n💾 Saving retriever...")
    import pickle
    with open("retriever.pkl", "wb") as f:
        pickle.dump(retriever, f)
    print("   ✅ Saved retriever.pkl")


if __name__ == "__main__":
    train()
