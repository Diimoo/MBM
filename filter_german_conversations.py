#!/usr/bin/env python3
"""
Filter German conversations from conversations.json and convert to Q&A format.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

INPUT_FILE = Path("conversations.json")
OUTPUT_FILE = Path("german_conversations_qa.jsonl")


def detect_language(text):
    """Simple language detection based on common words."""
    if not text or len(text) < 20:
        return "unknown"
    
    text_lower = text.lower()
    
    # German indicators
    german_words = [
        ' der ', ' die ', ' das ', ' und ', ' ist ', ' sind ', ' war ', ' hat ',
        ' ein ', ' eine ', ' für ', ' mit ', ' auf ', ' von ', ' zu ', ' den ',
        ' dem ', ' des ', ' nicht ', ' auch ', ' nach ', ' bei ', ' über ',
        ' kann ', ' wird ', ' wurde ', ' werden ', ' haben ', ' sein ',
        ' ich ', ' du ', ' er ', ' sie ', ' es ', ' wir ', ' ihr ',
        ' wenn ', ' dann ', ' aber ', ' oder ', ' weil ', ' dass ',
        ' diese ', ' dieser ', ' dieses ', ' welche ', ' welcher ',
        ' können ', ' möchten ', ' müssen ', ' sollen ', ' wollen ',
        'ä', 'ö', 'ü', 'ß',
    ]
    
    # English indicators
    english_words = [
        ' the ', ' is ', ' are ', ' was ', ' has ', ' have ', ' been ',
        ' a ', ' an ', ' for ', ' with ', ' on ', ' of ', ' to ',
        ' not ', ' also ', ' after ', ' at ', ' about ', ' can ',
        ' will ', ' would ', ' should ', ' could ', ' may ', ' might ',
        ' i ', ' you ', ' he ', ' she ', ' it ', ' we ', ' they ',
        ' if ', ' then ', ' but ', ' or ', ' because ', ' that ',
        ' this ', ' these ', ' which ', ' what ', ' who ', ' how ',
        ' here ', ' there ', ' where ', ' when ', ' why ',
    ]
    
    german_count = sum(1 for w in german_words if w in text_lower)
    english_count = sum(1 for w in english_words if w in text_lower)
    
    # Check for German umlauts (strong indicator)
    umlaut_count = sum(1 for c in text_lower if c in 'äöüß')
    german_count += umlaut_count * 2
    
    if german_count > english_count * 1.2:
        return "german"
    elif english_count > german_count * 1.2:
        return "english"
    else:
        return "unknown"


def extract_conversations(data):
    """Extract user-assistant message pairs from ChatGPT export format."""
    conversations = []
    
    for conv in data:
        title = conv.get("title", "")
        mapping = conv.get("mapping", {})
        
        # Build message chain
        messages = []
        for node_id, node in mapping.items():
            msg = node.get("message")
            if msg and msg.get("content"):
                content = msg["content"]
                if content.get("content_type") == "text":
                    parts = content.get("parts", [])
                    text = " ".join(str(p) for p in parts if p).strip()
                    if text and len(text) > 5:
                        role = msg.get("author", {}).get("role", "")
                        if role in ["user", "assistant"]:
                            messages.append({
                                "role": role,
                                "text": text,
                                "create_time": msg.get("create_time", 0)
                            })
        
        # Sort by time
        messages.sort(key=lambda x: x.get("create_time") or 0)
        
        # Extract Q&A pairs (user -> assistant)
        for i in range(len(messages) - 1):
            if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
                q = messages[i]["text"]
                a = messages[i+1]["text"]
                
                # Skip very long or very short
                if 10 < len(q) < 500 and 10 < len(a) < 1000:
                    conversations.append({
                        "question": q,
                        "answer": a,
                        "title": title
                    })
    
    return conversations


def main():
    print("=" * 60)
    print("🇩🇪 Filtering German Conversations")
    print("=" * 60)
    
    # Load JSON
    print("\n📥 Loading conversations.json...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"   Loaded {len(data)} conversations")
    
    # Extract Q&A pairs
    print("\n🔄 Extracting Q&A pairs...")
    all_pairs = extract_conversations(data)
    print(f"   Extracted {len(all_pairs)} Q&A pairs")
    
    # Filter by language
    print("\n🔍 Detecting language...")
    german_pairs = []
    english_pairs = []
    unknown_pairs = []
    
    for pair in all_pairs:
        # Check both question and answer
        q_lang = detect_language(pair["question"])
        a_lang = detect_language(pair["answer"])
        
        # Consider German if either is German and neither is clearly English
        if q_lang == "german" or a_lang == "german":
            if q_lang != "english" and a_lang != "english":
                german_pairs.append(pair)
            else:
                unknown_pairs.append(pair)
        elif q_lang == "english" or a_lang == "english":
            english_pairs.append(pair)
        else:
            unknown_pairs.append(pair)
    
    print(f"   German: {len(german_pairs)}")
    print(f"   English: {len(english_pairs)}")
    print(f"   Unknown: {len(unknown_pairs)}")
    
    # Also check unknown pairs more carefully
    print("\n🔄 Re-checking unknown pairs...")
    for pair in unknown_pairs:
        combined = pair["question"] + " " + pair["answer"]
        if any(c in combined for c in "äöüßÄÖÜ"):
            german_pairs.append(pair)
    
    print(f"   Final German count: {len(german_pairs)}")
    
    # Clean up pairs
    print("\n🧹 Cleaning pairs...")
    cleaned = []
    seen = set()
    
    for pair in german_pairs:
        q = pair["question"].strip()
        a = pair["answer"].strip()
        
        # Remove code blocks and markdown for cleaner Q&A
        q = re.sub(r'```[\s\S]*?```', '[code]', q)
        a = re.sub(r'```[\s\S]*?```', '[code]', a)
        
        # Skip if too much code
        if q.count('[code]') > 2 or a.count('[code]') > 2:
            continue
        
        # Skip duplicates
        key = (q[:100].lower(), a[:100].lower())
        if key in seen:
            continue
        seen.add(key)
        
        # Ensure question ends with ?
        if not q.endswith("?") and not q.endswith(".") and not q.endswith("!"):
            # Check if it's a question-like text
            q_words = ["was", "wer", "wo", "wann", "wie", "warum", "welche", "können", "kannst", "ist"]
            if any(q.lower().startswith(w) for w in q_words):
                q += "?"
        
        cleaned.append({
            "question": q,
            "answer": a,
            "context": ""
        })
    
    print(f"   Cleaned: {len(cleaned)} pairs")
    
    # Save
    print(f"\n💾 Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in cleaned:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    print(f"   ✅ Saved {len(cleaned)} German Q&A pairs")
    
    # Show samples
    print("\n" + "=" * 60)
    print("📝 Sample German Q&A pairs:")
    print("=" * 60)
    for i, p in enumerate(cleaned[:5]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Q: {p['question'][:100]}...")
        print(f"A: {p['answer'][:100]}...")
    
    return cleaned


if __name__ == "__main__":
    main()
