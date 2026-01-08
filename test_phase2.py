#!/usr/bin/env python3
"""Validate Phase 2 model with sentence-level embeddings."""

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import random
import re

from hierarchical_german_phase2 import (
    HierarchicalGermanPhase2, VOCAB_SIZE, text_to_indices, indices_to_text
)

def load_test_sentences(n=200):
    sentences = []
    try:
        ds = load_dataset("SkySyrup/tinystories_german", split="train", streaming=True)
        for item in tqdm(ds, desc="Loading", total=n * 3):
            text = item.get('text', '')
            if isinstance(text, str):
                for sent in re.split(r'(?<=[.!?])\s+', text):
                    sent = sent.strip()
                    if 10 <= len(sent) <= 100:
                        sentences.append(sent)
                        if len(sentences) >= n * 2:
                            break
            if len(sentences) >= n * 2:
                break
        sentences = random.sample(sentences, min(n, len(sentences)))
    except Exception as e:
        print(f"Error: {e}")
    return sentences

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = HierarchicalGermanPhase2().to(device)
    model.load_state_dict(torch.load("checkpoints/phase2_best.pth", map_location=device))
    model.eval()
    
    sentences = load_test_sentences(200)
    print(f"\nTesting on {len(sentences)} sentences\n")
    
    perfect = 0
    embeddings = []
    
    print("="*60)
    print("SAMPLE RECONSTRUCTIONS")
    print("="*60)
    
    with torch.no_grad():
        for i, sent in enumerate(sentences):
            chars = torch.tensor([text_to_indices(sent, 128)], device=device)
            outputs = model(chars)
            
            recon = indices_to_text(outputs['char_recon'].argmax(dim=-1)[0].cpu().tolist())
            if sent.strip() == recon[:len(sent.strip())].strip():
                perfect += 1
            
            embeddings.append(outputs['sent_emb'][0].cpu())
            
            if i < 10:
                print(f"In:  '{sent[:50]}'")
                print(f"Out: '{recon[:50]}'")
                print()
    
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"\n📝 Reconstruction: {perfect}/{len(sentences)} ({100*perfect/len(sentences):.1f}%)")
    
    # Compute pairwise similarities
    sims = []
    for i in range(min(50, len(embeddings))):
        for j in range(i+1, min(50, len(embeddings))):
            sim = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
            sims.append(sim)
    
    print(f"\n📊 Sentence Embedding Similarities ({len(sims)} pairs):")
    print(f"   Average: {sum(sims)/len(sims):.4f}")
    print(f"   Min: {min(sims):.4f}")
    print(f"   Max: {max(sims):.4f}")
    
    below_05 = sum(1 for s in sims if s < 0.5)
    below_07 = sum(1 for s in sims if s < 0.7)
    above_095 = sum(1 for s in sims if s >= 0.95)
    
    print(f"   Below 0.5: {below_05}/{len(sims)} ({100*below_05/len(sims):.1f}%)")
    print(f"   Below 0.7: {below_07}/{len(sims)} ({100*below_07/len(sims):.1f}%)")
    print(f"   Above 0.95: {above_095}/{len(sims)} ({100*above_095/len(sims):.1f}%)")
    
    if max(sims) < 0.95 and sum(sims)/len(sims) < 0.6:
        print("\n✅ PHASE 2 VALIDATED: Sentence embeddings well-differentiated!")
    elif max(sims) < 0.95:
        print("\n✅ PHASE 2 GOOD: No collapse detected")
    else:
        print("\n⚠️  WARNING: Some collapse detected")
    
    print("="*60)

if __name__ == "__main__":
    main()
