#!/usr/bin/env python3
"""
Download and prepare real German Q&A corpora for fine-tuning.
Sources: GermanQuAD, MLQA (German), XQuAD (German)
"""

import json
from datasets import load_dataset
from pathlib import Path

OUTPUT_FILE = Path("real_german_qa.jsonl")


def extract_qa_pairs(example):
    """Extract question-answer pairs from SQuAD-style format."""
    pairs = []
    question = example.get("question", "")
    
    # Get answer text
    answers = example.get("answers", {})
    if isinstance(answers, dict):
        answer_texts = answers.get("text", [])
    else:
        answer_texts = []
    
    # Get context for richer answers
    context = example.get("context", "")
    
    if question and answer_texts:
        # Use first answer
        answer = answer_texts[0] if answer_texts else ""
        if answer:
            pairs.append({
                "question": question.strip(),
                "answer": answer.strip(),
                "context": context[:500] if context else ""  # Truncate context
            })
    
    return pairs


def download_germanquad():
    """Download GermanQuAD dataset."""
    print("\n📥 Downloading GermanQuAD...")
    try:
        dataset = load_dataset("deepset/germanquad", split="train")
        pairs = []
        for example in dataset:
            pairs.extend(extract_qa_pairs(example))
        print(f"   ✅ GermanQuAD: {len(pairs)} Q&A pairs")
        return pairs
    except Exception as e:
        print(f"   ❌ GermanQuAD failed: {e}")
        return []


def download_mlqa_german():
    """Download MLQA German subset."""
    print("\n📥 Downloading MLQA (German)...")
    try:
        dataset = load_dataset("facebook/mlqa", "mlqa.de.de", split="test")
        pairs = []
        for example in dataset:
            pairs.extend(extract_qa_pairs(example))
        print(f"   ✅ MLQA German: {len(pairs)} Q&A pairs")
        return pairs
    except Exception as e:
        print(f"   ❌ MLQA failed: {e}")
        return []


def download_xquad_german():
    """Download XQuAD German subset."""
    print("\n📥 Downloading XQuAD (German)...")
    try:
        dataset = load_dataset("google/xquad", "xquad.de", split="validation")
        pairs = []
        for example in dataset:
            pairs.extend(extract_qa_pairs(example))
        print(f"   ✅ XQuAD German: {len(pairs)} Q&A pairs")
        return pairs
    except Exception as e:
        print(f"   ❌ XQuAD failed: {e}")
        return []


def download_german_dpr():
    """Download German DPR (if available)."""
    print("\n📥 Downloading GermanDPR...")
    try:
        dataset = load_dataset("deepset/germandpr", "german_dpr", split="train")
        pairs = []
        for example in dataset:
            question = example.get("question", "")
            # GermanDPR has positive_ctxs with answers
            pos_ctxs = example.get("positive_ctxs", [])
            if question and pos_ctxs:
                for ctx in pos_ctxs[:1]:  # Take first positive context
                    text = ctx.get("text", "")
                    if text:
                        # Use first sentence as answer approximation
                        answer = text.split(".")[0] + "." if "." in text else text[:200]
                        pairs.append({
                            "question": question.strip(),
                            "answer": answer.strip(),
                            "context": text[:500]
                        })
        print(f"   ✅ GermanDPR: {len(pairs)} Q&A pairs")
        return pairs
    except Exception as e:
        print(f"   ❌ GermanDPR failed: {e}")
        return []


def clean_and_filter(pairs):
    """Clean and filter Q&A pairs."""
    cleaned = []
    seen = set()
    
    for p in pairs:
        q = p["question"].strip()
        a = p["answer"].strip()
        
        # Skip empty or too short
        if len(q) < 5 or len(a) < 2:
            continue
        
        # Skip duplicates
        key = (q.lower(), a.lower())
        if key in seen:
            continue
        seen.add(key)
        
        # Ensure question ends with ?
        if not q.endswith("?"):
            q += "?"
        
        cleaned.append({
            "question": q,
            "answer": a,
            "context": p.get("context", "")
        })
    
    return cleaned


def main():
    print("=" * 60)
    print("🇩🇪 Real German Q&A Corpus Downloader")
    print("=" * 60)
    
    all_pairs = []
    
    # Download from multiple sources
    all_pairs.extend(download_germanquad())
    all_pairs.extend(download_mlqa_german())
    all_pairs.extend(download_xquad_german())
    all_pairs.extend(download_german_dpr())
    
    print("\n" + "=" * 60)
    print(f"📊 Total raw pairs: {len(all_pairs)}")
    
    # Clean and deduplicate
    cleaned = clean_and_filter(all_pairs)
    print(f"📊 After cleaning: {len(cleaned)}")
    
    # Save to JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in cleaned:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    print(f"\n✅ Saved to {OUTPUT_FILE}")
    
    # Show samples
    print("\n" + "=" * 60)
    print("📝 Sample Q&A pairs:")
    print("=" * 60)
    for i, p in enumerate(cleaned[:5]):
        print(f"\nQ: {p['question']}")
        print(f"A: {p['answer']}")
    
    return cleaned


if __name__ == "__main__":
    main()
