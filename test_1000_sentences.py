#!/usr/bin/env python3
"""
Large-scale validation test: 1000 random German sentences
Tests reconstruction, syllabification, and embedding differentiation.
"""

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import random
import re
import sys

sys.path.insert(0, '/home/ahmed/Downloads/Kandel')
from hierarchical_german_tabula_rasa import (
    HierarchicalGermanModel, VOCAB_SIZE, text_to_indices, indices_to_text
)


def load_test_sentences(n=1000):
    """Load n random German sentences from dataset."""
    print(f"Loading {n} random German sentences...")
    sentences = []
    
    try:
        ds = load_dataset("SkySyrup/tinystories_german", split="train", streaming=True)
        
        # Collect more than needed to allow random sampling
        all_sents = []
        for item in tqdm(ds, desc="Loading", total=n * 5):
            text = item.get('text', '')
            if isinstance(text, str):
                for sent in re.split(r'(?<=[.!?])\s+', text):
                    sent = sent.strip()
                    if 10 <= len(sent) <= 120 and re.search(r'[A-Za-zäöüÄÖÜß]', sent):
                        all_sents.append(sent)
                        if len(all_sents) >= n * 3:
                            break
            if len(all_sents) >= n * 3:
                break
        
        # Random sample
        sentences = random.sample(all_sents, min(n, len(all_sents)))
        
    except Exception as e:
        print(f"Error: {e}")
        return []
    
    print(f"Selected {len(sentences)} sentences for testing")
    return sentences


def test_model(model, sentences, device):
    """Run comprehensive test on sentences."""
    model.eval()
    
    results = {
        'total': len(sentences),
        'perfect_recon': 0,
        'partial_recon': 0,
        'failed_recon': 0,
        'similarities': [],
        'has_capitals': 0,
        'capitals_preserved': 0,
        'syllable_examples': [],
    }
    
    embeddings = []
    
    print("\n" + "="*70)
    print(f"TESTING ON {len(sentences)} RANDOM GERMAN SENTENCES")
    print("="*70)
    
    with torch.no_grad():
        for i, sent in enumerate(tqdm(sentences, desc="Testing")):
            chars = text_to_indices(sent, 128)
            indices = torch.tensor([chars], device=device)
            
            outputs = model(indices)
            
            # 1. Reconstruction quality
            recon_ids = outputs['char_recon'].argmax(dim=-1)[0]
            recon_text = indices_to_text(recon_ids.cpu().tolist())
            
            # Compare (ignore trailing spaces/padding)
            orig_clean = sent.strip()
            recon_clean = recon_text[:len(orig_clean)].strip()
            
            if orig_clean == recon_clean:
                results['perfect_recon'] += 1
            elif orig_clean.lower() == recon_clean.lower():
                results['partial_recon'] += 1
            else:
                results['failed_recon'] += 1
            
            # 2. Capital preservation
            capitals_in_orig = [c for c in sent if c.isupper()]
            if capitals_in_orig:
                results['has_capitals'] += 1
                capitals_in_recon = [c for c in recon_text[:len(sent)] if c.isupper()]
                if len(capitals_in_recon) >= len(capitals_in_orig) * 0.8:
                    results['capitals_preserved'] += 1
            
            # 3. Collect embeddings for similarity analysis
            morph_emb = outputs['morph_emb'][0, 1:len(sent)+1].mean(dim=0)
            embeddings.append(morph_emb)
            
            # 4. Sample syllabification
            if i < 20:  # First 20 examples
                syl_pred = (torch.sigmoid(outputs['syl_boundaries'][0]) > 0.5).int()
                marked = ""
                for j, c in enumerate(sent[:min(len(sent), 100)]):
                    marked += c
                    if j + 1 < len(syl_pred) and syl_pred[j + 1] == 1 and c not in ' .,!?':
                        marked += "·"
                results['syllable_examples'].append((sent[:50], marked[:60]))
    
    # Compute pairwise similarities (sample 100 pairs)
    print("\nComputing embedding similarities...")
    n_samples = min(100, len(embeddings))
    sample_indices = random.sample(range(len(embeddings)), n_samples)
    
    for i in range(n_samples):
        for j in range(i + 1, min(i + 10, n_samples)):  # Compare with next 10
            idx_i, idx_j = sample_indices[i], sample_indices[j]
            sim = F.cosine_similarity(
                embeddings[idx_i].unsqueeze(0),
                embeddings[idx_j].unsqueeze(0)
            ).item()
            results['similarities'].append(sim)
    
    return results


def print_results(results):
    """Print detailed results."""
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    # Reconstruction
    print("\n📝 RECONSTRUCTION ACCURACY")
    print(f"   Perfect: {results['perfect_recon']}/{results['total']} ({100*results['perfect_recon']/results['total']:.1f}%)")
    print(f"   Partial (case diff): {results['partial_recon']}/{results['total']} ({100*results['partial_recon']/results['total']:.1f}%)")
    print(f"   Failed: {results['failed_recon']}/{results['total']} ({100*results['failed_recon']/results['total']:.1f}%)")
    
    # Capitals
    print("\n🔠 CAPITAL PRESERVATION")
    if results['has_capitals'] > 0:
        print(f"   Sentences with capitals: {results['has_capitals']}")
        print(f"   Capitals preserved: {results['capitals_preserved']}/{results['has_capitals']} ({100*results['capitals_preserved']/results['has_capitals']:.1f}%)")
    
    # Embedding similarity
    print("\n📊 EMBEDDING DIFFERENTIATION")
    if results['similarities']:
        sims = results['similarities']
        avg_sim = sum(sims) / len(sims)
        min_sim = min(sims)
        max_sim = max(sims)
        
        print(f"   Pairwise similarities ({len(sims)} pairs):")
        print(f"   - Average: {avg_sim:.4f}")
        print(f"   - Min: {min_sim:.4f}")
        print(f"   - Max: {max_sim:.4f}")
        
        # Distribution
        below_07 = sum(1 for s in sims if s < 0.7)
        below_08 = sum(1 for s in sims if s < 0.8)
        below_09 = sum(1 for s in sims if s < 0.9)
        above_095 = sum(1 for s in sims if s >= 0.95)
        
        print(f"   - Below 0.7: {below_07}/{len(sims)} ({100*below_07/len(sims):.1f}%)")
        print(f"   - Below 0.8: {below_08}/{len(sims)} ({100*below_08/len(sims):.1f}%)")
        print(f"   - Below 0.9: {below_09}/{len(sims)} ({100*below_09/len(sims):.1f}%)")
        print(f"   - Above 0.95 (collapse indicator): {above_095}/{len(sims)} ({100*above_095/len(sims):.1f}%)")
        
        if max_sim < 0.95:
            print("\n   ✅ NO EMBEDDING COLLAPSE DETECTED")
        else:
            print(f"\n   ⚠️  WARNING: {above_095} pairs have similarity >= 0.95")
    
    # Syllabification examples
    print("\n📖 SYLLABIFICATION EXAMPLES")
    for orig, marked in results['syllable_examples'][:10]:
        print(f"   {marked}")
    
    print("\n" + "="*70)
    
    # Final verdict
    recon_rate = results['perfect_recon'] / results['total']
    avg_sim = sum(results['similarities']) / len(results['similarities']) if results['similarities'] else 1.0
    
    print("\n🏆 FINAL VERDICT")
    if recon_rate >= 0.95 and avg_sim < 0.85:
        print("   ✅ MODEL VALIDATED: High reconstruction + differentiated embeddings")
    elif recon_rate >= 0.90:
        print("   ✅ GOOD: Reconstruction working, embeddings acceptable")
    else:
        print("   ⚠️  NEEDS IMPROVEMENT")
    
    print("="*70)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    model = HierarchicalGermanModel(vocab_size=VOCAB_SIZE).to(device)
    
    checkpoint = "checkpoints/tabula_rasa_best.pth"
    print(f"Loading model from {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    
    # Load test sentences
    sentences = load_test_sentences(1000)
    
    if not sentences:
        print("Failed to load sentences!")
        return
    
    # Run test
    results = test_model(model, sentences, device)
    
    # Print results
    print_results(results)
    
    # Save results to file
    with open("logs/test_1000_results.txt", "w") as f:
        f.write("="*70 + "\n")
        f.write("1000 SENTENCE VALIDATION TEST RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total sentences: {results['total']}\n")
        f.write(f"Perfect reconstruction: {results['perfect_recon']} ({100*results['perfect_recon']/results['total']:.1f}%)\n")
        f.write(f"Capitals preserved: {results['capitals_preserved']}/{results['has_capitals']}\n")
        if results['similarities']:
            f.write(f"Average similarity: {sum(results['similarities'])/len(results['similarities']):.4f}\n")
            f.write(f"Max similarity: {max(results['similarities']):.4f}\n")
        f.write("\nSyllabification examples:\n")
        for orig, marked in results['syllable_examples']:
            f.write(f"  {marked}\n")
    
    print(f"\nResults saved to logs/test_1000_results.txt")


if __name__ == "__main__":
    main()
