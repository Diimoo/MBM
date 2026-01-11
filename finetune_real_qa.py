#!/usr/bin/env python3
"""
Fine-tune the hierarchical German Q&A model on real German Q&A data.
Uses XQuAD German dataset for semantically meaningful Q&A pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import random
from tqdm import tqdm

# Import the model
from hierarchical_german_phase6_qa import (
    HierarchicalGermanQA, char_to_idx, idx_to_char, VOCAB_SIZE,
    PAD_TOKEN, Q_TOKEN, A_TOKEN, EOS_TOKEN,
    text_to_indices, answer_to_indices, indices_to_text
)


class CharVocab:
    """Simple character vocabulary wrapper."""
    def __init__(self):
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL_QA_FILE = Path("real_german_qa.jsonl")
CHECKPOINT_PATH = Path("checkpoints/phase6_qa_best.pth")
FINETUNE_CHECKPOINT = Path("checkpoints/phase6_qa_finetuned.pth")


class RealQADataset(Dataset):
    """Dataset for real German Q&A pairs."""
    
    def __init__(self, qa_pairs, vocab, max_len=128):  # Must match model's max_len
        self.qa_pairs = qa_pairs
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        pair = self.qa_pairs[idx]
        question = pair["question"]
        answer = pair["answer"]
        
        # Encode question (input) and answer (target) using model's encoding functions
        q_indices = text_to_indices(question, self.max_len)
        a_indices = answer_to_indices(answer, self.max_len)
        
        return {
            "q_chars": torch.tensor(q_indices, dtype=torch.long),
            "a_chars": torch.tensor(a_indices, dtype=torch.long),
            "question": question,
            "answer": answer
        }


def load_real_qa_data():
    """Load real German Q&A pairs."""
    pairs = []
    with open(REAL_QA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            pair = json.loads(line.strip())
            # Filter: skip very short answers or questions
            if len(pair["question"]) >= 10 and len(pair["answer"]) >= 2:
                pairs.append(pair)
    return pairs


def create_answer_with_context(question, answer, context=""):
    """Create a more natural answer using context."""
    q_lower = question.lower()
    
    # For "Wie viele" questions, create full sentence answers
    if q_lower.startswith("wie viele"):
        # Extract what we're counting
        if "punkte" in q_lower:
            return f"Es waren {answer} Punkte."
        elif "sacks" in q_lower:
            return f"Es waren {answer} Sacks."
        elif "tackles" in q_lower:
            return f"Es wurden {answer} Tackles registriert."
        else:
            return f"Es sind {answer}."
    
    # For "Wer" questions
    if q_lower.startswith("wer"):
        return f"{answer}."
    
    # For "Was" questions
    if q_lower.startswith("was"):
        return f"Das ist {answer}."
    
    # For "Wo" questions
    if q_lower.startswith("wo"):
        return f"In {answer}." if not answer.startswith("in") else f"{answer}."
    
    # For "Wann" questions
    if q_lower.startswith("wann"):
        return f"Am {answer}." if not any(answer.startswith(x) for x in ["am", "im", "um"]) else f"{answer}."
    
    # Default: return as-is
    return answer


def augment_qa_pairs(pairs):
    """Augment Q&A pairs with better formatted answers."""
    augmented = []
    for pair in pairs:
        q = pair["question"]
        a = pair["answer"]
        ctx = pair.get("context", "")
        
        # Create natural answer
        natural_answer = create_answer_with_context(q, a, ctx)
        
        augmented.append({
            "question": q,
            "answer": natural_answer,
            "context": ctx
        })
        
        # Also keep original short answer for variety
        if natural_answer != a:
            augmented.append({
                "question": q,
                "answer": a,
                "context": ctx
            })
    
    return augmented


def test_qa(model, vocab):
    """Test Q&A capabilities using model's built-in answer_question method."""
    model.eval()
    
    print("\n" + "=" * 60)
    print("Q&A TEST (Real German)")
    print("=" * 60)
    
    # Test questions from real data
    test_questions = [
        "Wie viele Punkte erzielte das Team?",
        "Wer gewann das Spiel?",
        "Wo fand das Turnier statt?",
        "Was ist die Hauptstadt von Deutschland?",
        "Wann wurde die Universität gegründet?",
    ]
    
    print("\n💬 Answer Generation:")
    for q in test_questions:
        try:
            answer = model.answer_question(q, temperature=0.7)
            print(f"  Q: {q}")
            print(f"  A: {answer}\n")
        except Exception as e:
            print(f"  Q: {q}")
            print(f"  A: [Error: {e}]\n")


def finetune():
    """Fine-tune the model on real German Q&A data."""
    print("=" * 60)
    print("🇩🇪 Fine-tuning on Real German Q&A Data")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # Load real Q&A data
    print("\n📥 Loading real Q&A data...")
    pairs = load_real_qa_data()
    print(f"   Loaded {len(pairs)} Q&A pairs")
    
    # Augment with better formatted answers
    pairs = augment_qa_pairs(pairs)
    print(f"   After augmentation: {len(pairs)} pairs")
    
    # Shuffle and split
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.9)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    print(f"   Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    # Load pretrained model
    print("\n📦 Loading pretrained model...")
    
    # Create vocab (uses global vocab from phase6 module)
    vocab = CharVocab()
    print(f"   Vocab size: {len(vocab.char_to_idx)}")
    
    # Create model and load weights
    model = HierarchicalGermanQA(VOCAB_SIZE).to(DEVICE)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    # Remove buffer keys that may cause issues
    state_dict = {k: v for k, v in state_dict.items() if 'causal_mask' not in k}
    model.load_state_dict(state_dict, strict=False)
    print(f"   ✅ Loaded from {CHECKPOINT_PATH}")
    
    # Create datasets
    train_dataset = RealQADataset(train_pairs, vocab)
    val_dataset = RealQADataset(val_pairs, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Fine-tuning settings (lower LR for fine-tuning)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.char_to_idx.get("<pad>", 0))
    
    # Training loop
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0
    
    print("\n🚀 Starting fine-tuning...")
    for epoch in range(10):
        # Training
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/10")
        for batch in pbar:
            q_chars = batch["q_chars"].to(DEVICE)
            a_chars = batch["a_chars"].to(DEVICE)
            
            optimizer.zero_grad()
            # Forward pass: question chars as input, answer chars as target for generation
            outputs = model(q_chars, target_chars=a_chars[:, :-1], mode=1)
            
            # Generation loss using gen_logits
            gen_logits = outputs["gen_logits"]
            targets = a_chars[:, 1:]  # Shift by 1 for next-token prediction
            loss = criterion(gen_logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
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
        
        print(f"\nEpoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Test Q&A
        test_qa(model, vocab)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab": {"char_to_idx": vocab.char_to_idx},
                "val_loss": val_loss,
                "epoch": epoch + 1
            }, FINETUNE_CHECKPOINT)
            print(f"💾 Saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                break
    
    print("\n✅ Fine-tuning complete!")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Checkpoint: {FINETUNE_CHECKPOINT}")
    
    # Final demo
    print("\n" + "=" * 60)
    print("FINAL Q&A DEMO")
    print("=" * 60)
    test_qa(model, vocab)


if __name__ == "__main__":
    finetune()
