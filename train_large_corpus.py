#!/usr/bin/env python3
"""
Train German Q&A model on large combined corpus WITHOUT RAG.
Model learns directly from data - no frozen weights or retrieval augmentation.
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

from hierarchical_german_phase6_qa import (
    HierarchicalGermanQA, char_to_idx, idx_to_char, VOCAB_SIZE,
    PAD_TOKEN, Q_TOKEN, A_TOKEN, EOS_TOKEN,
    text_to_indices, answer_to_indices, indices_to_text
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = Path("checkpoints/phase6_qa_best.pth")
OUTPUT_CHECKPOINT = Path("checkpoints/phase6_qa_large.pth")


class QADataset(Dataset):
    """Simple Q&A dataset - model learns everything, no RAG."""
    
    def __init__(self, qa_pairs, max_len=128):
        self.qa_pairs = qa_pairs
        self.max_len = max_len
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        pair = self.qa_pairs[idx]
        question = pair["question"]
        answer = pair["answer"]
        
        # Encode using model's native encoding
        q_indices = text_to_indices(question, self.max_len)
        a_indices = answer_to_indices(answer, self.max_len)
        
        return {
            "q_chars": torch.tensor(q_indices, dtype=torch.long),
            "a_chars": torch.tensor(a_indices, dtype=torch.long),
        }


def load_all_corpora():
    """Load and combine all available German Q&A corpora."""
    all_pairs = []
    
    # 1. Real German Q&A (XQuAD)
    if Path("real_german_qa.jsonl").exists():
        with open("real_german_qa.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                all_pairs.append(json.loads(line.strip()))
        print(f"   📚 Real Q&A: {len(all_pairs)} pairs")
    
    # 2. German conversations (filtered from user's conversations.json)
    conv_count = 0
    if Path("german_conversations_qa.jsonl").exists():
        with open("german_conversations_qa.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                all_pairs.append(json.loads(line.strip()))
                conv_count += 1
        print(f"   💬 Conversations: {conv_count} pairs")
    
    # 3. Synthetic German Q&A (sample to balance)
    synth_count = 0
    if Path("german_qa_dataset.jsonl").exists():
        synth_pairs = []
        with open("german_qa_dataset.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                synth_pairs.append({
                    "question": data["question"],
                    "answer": data["answer"],
                    "context": ""
                })
        # Sample to avoid overwhelming with synthetic
        random.shuffle(synth_pairs)
        sample_size = min(30000, len(synth_pairs))
        all_pairs.extend(synth_pairs[:sample_size])
        synth_count = sample_size
        print(f"   🔧 Synthetic: {synth_count} pairs (sampled)")
    
    # 4. Combined corpus if exists
    if Path("combined_german_qa.jsonl").exists():
        combined_count = 0
        seen = set((p["question"][:50], p["answer"][:50]) for p in all_pairs)
        with open("combined_german_qa.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                p = json.loads(line.strip())
                key = (p["question"][:50], p["answer"][:50])
                if key not in seen:
                    all_pairs.append(p)
                    seen.add(key)
                    combined_count += 1
        if combined_count > 0:
            print(f"   📦 Combined (new): {combined_count} pairs")
    
    return all_pairs


def deduplicate(pairs):
    """Remove duplicate Q&A pairs."""
    seen = set()
    unique = []
    for p in pairs:
        key = (p["question"].lower().strip()[:80], p["answer"].lower().strip()[:80])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def test_qa(model):
    """Test Q&A generation."""
    model.eval()
    
    print("\n" + "=" * 60)
    print("Q&A TEST")
    print("=" * 60)
    
    test_questions = [
        "Was ist die Hauptstadt von Deutschland?",
        "Wann fiel die Berliner Mauer?",
        "Wer schrieb Faust?",
        "Wie viele Bundesländer hat Deutschland?",
        "Was bedeutet das Wort Gemütlichkeit?",
        "Wie geht es dir?",
    ]
    
    print("\n💬 Answers:")
    for q in test_questions:
        try:
            answer = model.answer_question(q, temperature=0.7)
            # Clean
            if "<EOS>" in answer:
                answer = answer[:answer.index("<EOS>")]
            if "<PAD>" in answer:
                answer = answer[:answer.index("<PAD>")]
            print(f"  Q: {q}")
            print(f"  A: {answer.strip()}\n")
        except Exception as e:
            print(f"  Q: {q}")
            print(f"  A: [Error: {e}]\n")


def train():
    """Train on large corpus without RAG."""
    print("=" * 60)
    print("🇩🇪 Training on Large German Q&A Corpus (No RAG)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # Load all corpora
    print("\n📥 Loading all corpora...")
    pairs = load_all_corpora()
    print(f"   Total raw: {len(pairs)}")
    
    # Deduplicate
    pairs = deduplicate(pairs)
    print(f"   After dedup: {len(pairs)}")
    
    # Shuffle and split
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.95)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    print(f"   Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    # Datasets
    train_dataset = QADataset(train_pairs)
    val_dataset = QADataset(val_pairs)
    
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # Cosine annealing with warm restarts
    num_epochs = 30
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-5
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    
    # Training
    best_val_loss = float("inf")
    patience = 7
    patience_counter = 0
    
    print(f"\n🚀 Training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Train
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
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        train_loss /= len(train_loader)
        
        # Validate
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
        
        print(f"\nEpoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, LR={optimizer.param_groups[0]['lr']:.2e}")
        
        # Test every 5 epochs
        if (epoch + 1) % 5 == 0:
            test_qa(model)
        
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
    test_qa(model)


if __name__ == "__main__":
    train()
