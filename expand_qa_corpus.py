#!/usr/bin/env python3
"""
Expand German Q&A corpus with multiple sources and augmentation.
"""

import json
import random
from pathlib import Path
from datasets import load_dataset

OUTPUT_FILE = Path("combined_german_qa.jsonl")


def load_xquad_german():
    """Load XQuAD German dataset."""
    print("\n📥 Loading XQuAD German...")
    try:
        ds = load_dataset("google/xquad", "xquad.de", split="validation")
        pairs = []
        for ex in ds:
            q = ex.get("question", "").strip()
            answers = ex.get("answers", {})
            a_texts = answers.get("text", []) if isinstance(answers, dict) else []
            ctx = ex.get("context", "")[:500]
            if q and a_texts:
                pairs.append({
                    "question": q if q.endswith("?") else q + "?",
                    "answer": a_texts[0].strip(),
                    "context": ctx
                })
        print(f"   ✅ XQuAD: {len(pairs)} pairs")
        return pairs
    except Exception as e:
        print(f"   ❌ XQuAD failed: {e}")
        return []


def load_synthetic_qa():
    """Load our synthetic German Q&A dataset."""
    print("\n📥 Loading synthetic German Q&A...")
    pairs = []
    try:
        with open("german_qa_dataset.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                pairs.append({
                    "question": data["question"],
                    "answer": data["answer"],
                    "context": ""
                })
        print(f"   ✅ Synthetic: {len(pairs)} pairs")
        return pairs
    except Exception as e:
        print(f"   ❌ Synthetic failed: {e}")
        return []


def load_real_qa():
    """Load previously downloaded real Q&A."""
    print("\n📥 Loading real German Q&A...")
    pairs = []
    try:
        with open("real_german_qa.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                pairs.append(json.loads(line.strip()))
        print(f"   ✅ Real Q&A: {len(pairs)} pairs")
        return pairs
    except Exception as e:
        print(f"   ❌ Real Q&A failed: {e}")
        return []


def create_german_commonsense_qa():
    """Create German commonsense Q&A pairs."""
    print("\n📥 Creating German commonsense Q&A...")
    
    qa_pairs = [
        # Geography
        ("Was ist die Hauptstadt von Deutschland?", "Berlin ist die Hauptstadt von Deutschland."),
        ("Welcher Fluss fließt durch Berlin?", "Die Spree fließt durch Berlin."),
        ("Wie viele Bundesländer hat Deutschland?", "Deutschland hat 16 Bundesländer."),
        ("Welches ist das größte Bundesland?", "Bayern ist das größte Bundesland."),
        ("An welche Länder grenzt Deutschland?", "Deutschland grenzt an neun Länder."),
        ("Welches Gebirge liegt im Süden Deutschlands?", "Die Alpen liegen im Süden Deutschlands."),
        ("Wie heißt der höchste Berg Deutschlands?", "Die Zugspitze ist der höchste Berg."),
        ("Welche Stadt ist für ihr Oktoberfest bekannt?", "München ist für das Oktoberfest bekannt."),
        ("Wo befindet sich der Kölner Dom?", "Der Kölner Dom befindet sich in Köln."),
        ("Welcher See ist der größte in Deutschland?", "Der Bodensee ist der größte See."),
        
        # History
        ("Wann fiel die Berliner Mauer?", "Die Berliner Mauer fiel am 9. November 1989."),
        ("Wann wurde Deutschland wiedervereinigt?", "Deutschland wurde 1990 wiedervereinigt."),
        ("Wer war der erste Bundeskanzler?", "Konrad Adenauer war der erste Bundeskanzler."),
        ("Wann begann der Zweite Weltkrieg?", "Der Zweite Weltkrieg begann 1939."),
        ("Wann endete der Zweite Weltkrieg?", "Der Zweite Weltkrieg endete 1945."),
        
        # Culture
        ("Wer schrieb Faust?", "Johann Wolfgang von Goethe schrieb Faust."),
        ("Welcher Komponist war taub?", "Ludwig van Beethoven wurde taub."),
        ("Wer erfand den Buchdruck?", "Johannes Gutenberg erfand den Buchdruck."),
        ("Welches Märchen handelt von Rotkäppchen?", "Es ist ein Märchen der Brüder Grimm."),
        
        # Science
        ("Wer entwickelte die Relativitätstheorie?", "Albert Einstein entwickelte die Relativitätstheorie."),
        ("Was ist die Formel für Wasser?", "Die chemische Formel für Wasser ist H2O."),
        ("Wie viele Planeten hat unser Sonnensystem?", "Unser Sonnensystem hat acht Planeten."),
        ("Was ist der größte Planet?", "Jupiter ist der größte Planet."),
        ("Wie heißt unser nächster Stern?", "Die Sonne ist unser nächster Stern."),
        
        # Daily life
        ("Wie viele Tage hat ein Jahr?", "Ein Jahr hat 365 oder 366 Tage."),
        ("Wie viele Monate hat ein Jahr?", "Ein Jahr hat zwölf Monate."),
        ("Wie viele Stunden hat ein Tag?", "Ein Tag hat 24 Stunden."),
        ("Was ist die Währung in Deutschland?", "Der Euro ist die Währung in Deutschland."),
        ("Welche Farben hat die deutsche Flagge?", "Schwarz, Rot und Gold."),
        
        # Animals
        ("Welches Tier ist das schnellste?", "Der Gepard ist das schnellste Landtier."),
        ("Welches Tier ist das größte?", "Der Blauwal ist das größte Tier."),
        ("Wie viele Beine hat eine Spinne?", "Eine Spinne hat acht Beine."),
        ("Können Pinguine fliegen?", "Nein, Pinguine können nicht fliegen."),
        
        # Food
        ("Was ist Sauerkraut?", "Sauerkraut ist fermentierter Kohl."),
        ("Was ist eine Brezel?", "Eine Brezel ist ein Laugengebäck."),
        ("Woraus wird Bier gebraut?", "Bier wird aus Wasser, Malz, Hopfen und Hefe gebraut."),
    ]
    
    pairs = [{"question": q, "answer": a, "context": ""} for q, a in qa_pairs]
    
    # Augment with variations
    augmented = []
    for p in pairs:
        augmented.append(p)
        # Add informal version
        q = p["question"]
        if q.startswith("Was ist"):
            augmented.append({
                "question": q.replace("Was ist", "Was bedeutet"),
                "answer": p["answer"],
                "context": ""
            })
    
    print(f"   ✅ Commonsense: {len(augmented)} pairs")
    return augmented


def create_conversational_qa():
    """Create conversational German Q&A pairs."""
    print("\n📥 Creating conversational Q&A...")
    
    qa_pairs = [
        # Greetings
        ("Wie geht es dir?", "Mir geht es gut, danke der Nachfrage."),
        ("Wie heißt du?", "Ich bin ein KI-Assistent für deutsche Sprache."),
        ("Was kannst du?", "Ich kann Fragen auf Deutsch beantworten."),
        ("Woher kommst du?", "Ich wurde für die deutsche Sprache entwickelt."),
        ("Sprichst du Deutsch?", "Ja, ich spreche Deutsch."),
        
        # Opinions
        ("Was denkst du über das Wetter?", "Das Wetter ist ein interessantes Thema."),
        ("Magst du Musik?", "Musik ist eine wunderbare Kunstform."),
        ("Was ist dein Lieblingsessen?", "Als KI habe ich keine Vorlieben."),
        
        # Help
        ("Kannst du mir helfen?", "Ja, ich helfe gerne. Was möchtest du wissen?"),
        ("Ich verstehe nicht.", "Kein Problem, ich erkläre es nochmal."),
        ("Das ist schwer.", "Lass uns das zusammen durchgehen."),
    ]
    
    pairs = [{"question": q, "answer": a, "context": ""} for q, a in qa_pairs]
    print(f"   ✅ Conversational: {len(pairs)} pairs")
    return pairs


def deduplicate(pairs):
    """Remove duplicate Q&A pairs."""
    seen = set()
    unique = []
    for p in pairs:
        key = (p["question"].lower().strip(), p["answer"].lower().strip()[:50])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def main():
    print("=" * 60)
    print("🇩🇪 Expanding German Q&A Corpus")
    print("=" * 60)
    
    all_pairs = []
    
    # Load from multiple sources
    all_pairs.extend(load_xquad_german())
    all_pairs.extend(load_real_qa())
    all_pairs.extend(create_german_commonsense_qa())
    all_pairs.extend(create_conversational_qa())
    
    # Sample from synthetic (too large to use all)
    synthetic = load_synthetic_qa()
    if synthetic:
        # Take 20k diverse samples from synthetic
        random.shuffle(synthetic)
        all_pairs.extend(synthetic[:20000])
    
    print("\n" + "=" * 60)
    print(f"📊 Total raw pairs: {len(all_pairs)}")
    
    # Deduplicate
    unique = deduplicate(all_pairs)
    print(f"📊 After dedup: {len(unique)}")
    
    # Shuffle
    random.shuffle(unique)
    
    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for p in unique:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"\n✅ Saved to {OUTPUT_FILE}")
    
    # Stats
    with_context = sum(1 for p in unique if p.get("context"))
    print(f"\n📈 Statistics:")
    print(f"   Total pairs: {len(unique)}")
    print(f"   With context: {with_context}")
    print(f"   Without context: {len(unique) - with_context}")


if __name__ == "__main__":
    main()
