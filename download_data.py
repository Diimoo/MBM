#!/usr/bin/env python3
"""Download WikiText-103 and German data for training."""

from datasets import load_dataset
from tqdm import tqdm
import os

def main():
    print("=" * 60)
    print("DOWNLOADING TRAINING DATA")
    print("=" * 60)
    
    # 1. WikiText-103 (English)
    print("\n[1/2] Downloading WikiText-103...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    
    en_sentences = []
    for item in tqdm(dataset, desc="Processing English"):
        text = item['text'].strip()
        # Filter: 20-150 chars, no headers (start with =)
        if 20 <= len(text) <= 150 and not text.startswith('='):
            en_sentences.append(text)
    
    with open('wikitext_en.txt', 'w', encoding='utf-8') as f:
        for sent in en_sentences:
            f.write(sent + '\n')
    
    print(f"✓ English sentences: {len(en_sentences)}")
    
    # 2. German Wikipedia (smaller, for cross-lingual)
    print("\n[2/2] Downloading German Wikipedia...")
    try:
        de_dataset = load_dataset("wikipedia", "20220301.de", split="train[:50000]")
        
        de_sentences = []
        for item in tqdm(de_dataset, desc="Processing German"):
            text = item['text']
            # Split into sentences and filter
            for line in text.split('\n'):
                line = line.strip()
                if 20 <= len(line) <= 150:
                    de_sentences.append(line)
                    if len(de_sentences) >= 100000:
                        break
            if len(de_sentences) >= 100000:
                break
        
        with open('wiki_de.txt', 'w', encoding='utf-8') as f:
            for sent in de_sentences[:100000]:
                f.write(sent + '\n')
        
        print(f"✓ German sentences: {len(de_sentences)}")
    except Exception as e:
        print(f"⚠️ German download failed: {e}")
        print("Creating minimal German dataset from templates...")
        
        # Fallback: expand existing German templates
        de_templates = [
            "die {noun} {verb} {adverb}",
            "ein {adj} {noun} {verb} im {place}",
            "{noun} {verb} immer {adverb}",
            "der {adj} {noun} ist {state}",
        ]
        nouns = ["katze", "hund", "vogel", "baum", "haus", "auto", "kind", "mann", "frau"]
        verbs = ["läuft", "springt", "schläft", "arbeitet", "spielt", "lernt", "denkt"]
        adverbs = ["schnell", "langsam", "leise", "laut", "gut", "schlecht"]
        adjs = ["groß", "klein", "alt", "neu", "schön", "hell", "dunkel"]
        places = ["garten", "haus", "wald", "park", "zimmer", "büro"]
        states = ["müde", "glücklich", "traurig", "hungrig", "ruhig"]
        
        import random
        de_sentences = []
        for _ in range(10000):
            template = random.choice(de_templates)
            sent = template.format(
                noun=random.choice(nouns),
                verb=random.choice(verbs),
                adverb=random.choice(adverbs),
                adj=random.choice(adjs),
                place=random.choice(places),
                state=random.choice(states)
            )
            de_sentences.append(sent)
        
        with open('wiki_de.txt', 'w', encoding='utf-8') as f:
            for sent in de_sentences:
                f.write(sent + '\n')
        
        print(f"✓ German sentences (templates): {len(de_sentences)}")
    
    print("\n" + "=" * 60)
    print("✅ DATA DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"English: wikitext_en.txt ({os.path.getsize('wikitext_en.txt') / 1e6:.1f} MB)")
    print(f"German:  wiki_de.txt ({os.path.getsize('wiki_de.txt') / 1e6:.1f} MB)")

if __name__ == '__main__':
    main()
