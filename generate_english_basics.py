#!/usr/bin/env python3
"""
Generate English basics training data for hierarchical model.
Level 0: Character sequences
Level 1: Syllable-annotated words
Level 2: Morpheme-annotated words
Level 3: Word sequences (sentences)
"""

import json
import random
from pathlib import Path

OUTPUT_DIR = Path("english_basics_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# ENGLISH SYLLABLE DICTIONARY (common words with syllable breaks)
# =============================================================================

SYLLABLE_DICT = {
    # 1-syllable words
    "cat": ["cat"], "dog": ["dog"], "run": ["run"], "jump": ["jump"],
    "walk": ["walk"], "talk": ["talk"], "book": ["book"], "look": ["look"],
    "make": ["make"], "take": ["take"], "give": ["give"], "live": ["live"],
    "house": ["house"], "mouse": ["mouse"], "tree": ["tree"], "free": ["free"],
    "big": ["big"], "small": ["small"], "tall": ["tall"], "fall": ["fall"],
    "red": ["red"], "blue": ["blue"], "green": ["green"], "white": ["white"],
    "black": ["black"], "brown": ["brown"], "pink": ["pink"], "gray": ["gray"],
    "sun": ["sun"], "moon": ["moon"], "star": ["star"], "sky": ["sky"],
    "day": ["day"], "night": ["night"], "light": ["light"], "dark": ["dark"],
    "hot": ["hot"], "cold": ["cold"], "warm": ["warm"], "cool": ["cool"],
    "fast": ["fast"], "slow": ["slow"], "new": ["new"], "old": ["old"],
    "good": ["good"], "bad": ["bad"], "great": ["great"], "nice": ["nice"],
    "hand": ["hand"], "foot": ["foot"], "head": ["head"], "eye": ["eye"],
    "ear": ["ear"], "nose": ["nose"], "mouth": ["mouth"], "face": ["face"],
    "man": ["man"], "men": ["men"], "child": ["child"], "boy": ["boy"],
    "girl": ["girl"], "friend": ["friend"], "love": ["love"], "hate": ["hate"],
    "work": ["work"], "play": ["play"], "read": ["read"], "write": ["write"],
    "think": ["think"], "know": ["know"], "see": ["see"], "hear": ["hear"],
    "feel": ["feel"], "want": ["want"], "need": ["need"], "like": ["like"],
    "come": ["come"], "go": ["go"], "get": ["get"], "put": ["put"],
    "say": ["say"], "tell": ["tell"], "ask": ["ask"], "find": ["find"],
    "use": ["use"], "help": ["help"], "try": ["try"], "call": ["call"],
    "keep": ["keep"], "let": ["let"], "seem": ["seem"], "leave": ["leave"],
    "time": ["time"], "year": ["year"], "way": ["way"], "thing": ["thing"],
    "world": ["world"], "life": ["life"], "part": ["part"], "place": ["place"],
    
    # 2-syllable words
    "happy": ["hap", "py"], "sunny": ["sun", "ny"], "funny": ["fun", "ny"],
    "pretty": ["pret", "ty"], "little": ["lit", "tle"], "middle": ["mid", "dle"],
    "water": ["wa", "ter"], "paper": ["pa", "per"], "mother": ["moth", "er"],
    "father": ["fa", "ther"], "sister": ["sis", "ter"], "brother": ["broth", "er"],
    "children": ["chil", "dren"], "people": ["peo", "ple"], "woman": ["wom", "an"],
    "country": ["coun", "try"], "city": ["cit", "y"], "party": ["par", "ty"],
    "money": ["mon", "ey"], "story": ["sto", "ry"], "study": ["stud", "y"],
    "music": ["mu", "sic"], "movie": ["mov", "ie"], "picture": ["pic", "ture"],
    "garden": ["gar", "den"], "window": ["win", "dow"], "number": ["num", "ber"],
    "problem": ["prob", "lem"], "answer": ["an", "swer"], "question": ["ques", "tion"],
    "reason": ["rea", "son"], "person": ["per", "son"], "season": ["sea", "son"],
    "morning": ["morn", "ing"], "evening": ["eve", "ning"], "something": ["some", "thing"],
    "nothing": ["noth", "ing"], "anything": ["an", "y", "thing"], "everything": ["ev", "ry", "thing"],
    "today": ["to", "day"], "tonight": ["to", "night"], "away": ["a", "way"],
    "around": ["a", "round"], "about": ["a", "bout"], "again": ["a", "gain"],
    "against": ["a", "gainst"], "along": ["a", "long"], "across": ["a", "cross"],
    "among": ["a", "mong"], "between": ["be", "tween"], "before": ["be", "fore"],
    "behind": ["be", "hind"], "below": ["be", "low"], "beside": ["be", "side"],
    "under": ["un", "der"], "over": ["o", "ver"], "after": ["af", "ter"],
    "begin": ["be", "gin"], "become": ["be", "come"], "believe": ["be", "lieve"],
    "forget": ["for", "get"], "remember": ["re", "mem", "ber"],
    "yellow": ["yel", "low"], "orange": ["or", "ange"], "purple": ["pur", "ple"],
    "silver": ["sil", "ver"], "golden": ["gold", "en"], "wooden": ["wood", "en"],
    "public": ["pub", "lic"], "private": ["pri", "vate"], "simple": ["sim", "ple"],
    "double": ["dou", "ble"], "single": ["sin", "gle"], "final": ["fi", "nal"],
    "local": ["lo", "cal"], "central": ["cen", "tral"], "special": ["spe", "cial"],
    "natural": ["nat", "u", "ral"], "normal": ["nor", "mal"], "human": ["hu", "man"],
    "office": ["of", "fice"], "service": ["ser", "vice"], "power": ["pow", "er"],
    "minute": ["min", "ute"], "second": ["sec", "ond"], "moment": ["mo", "ment"],
    "level": ["lev", "el"], "system": ["sys", "tem"], "program": ["pro", "gram"],
    
    # 3-syllable words
    "beautiful": ["beau", "ti", "ful"], "wonderful": ["won", "der", "ful"],
    "different": ["dif", "fer", "ent"], "important": ["im", "por", "tant"],
    "possible": ["pos", "si", "ble"], "impossible": ["im", "pos", "si", "ble"],
    "interest": ["in", "ter", "est"], "government": ["gov", "ern", "ment"],
    "together": ["to", "geth", "er"], "another": ["an", "oth", "er"],
    "company": ["com", "pa", "ny"], "family": ["fam", "i", "ly"],
    "history": ["his", "to", "ry"], "industry": ["in", "dus", "try"],
    "example": ["ex", "am", "ple"], "exercise": ["ex", "er", "cise"],
    "however": ["how", "ev", "er"], "whatever": ["what", "ev", "er"],
    "whenever": ["when", "ev", "er"], "wherever": ["wher", "ev", "er"],
    "computer": ["com", "pu", "ter"], "september": ["sep", "tem", "ber"],
    "november": ["no", "vem", "ber"], "december": ["de", "cem", "ber"],
    "animal": ["an", "i", "mal"], "hospital": ["hos", "pi", "tal"],
    "customer": ["cus", "tom", "er"], "telephone": ["tel", "e", "phone"],
    "tomorrow": ["to", "mor", "row"], "yesterday": ["yes", "ter", "day"],
    "afternoon": ["af", "ter", "noon"], "everything": ["ev", "ry", "thing"],
    "anywhere": ["an", "y", "where"], "everyone": ["ev", "ry", "one"],
    "nobody": ["no", "bod", "y"], "somebody": ["some", "bod", "y"],
    "absolute": ["ab", "so", "lute"], "adventure": ["ad", "ven", "ture"],
    "dangerous": ["dan", "ger", "ous"], "delicious": ["de", "li", "cious"],
    "attention": ["at", "ten", "tion"], "condition": ["con", "di", "tion"],
    "direction": ["di", "rec", "tion"], "education": ["ed", "u", "ca", "tion"],
    "collection": ["col", "lec", "tion"], "connection": ["con", "nec", "tion"],
    "protection": ["pro", "tec", "tion"], "production": ["pro", "duc", "tion"],
    
    # 4+ syllable words
    "information": ["in", "for", "ma", "tion"],
    "international": ["in", "ter", "na", "tion", "al"],
    "university": ["u", "ni", "ver", "si", "ty"],
    "opportunity": ["op", "por", "tu", "ni", "ty"],
    "communication": ["com", "mu", "ni", "ca", "tion"],
    "organization": ["or", "gan", "i", "za", "tion"],
    "responsibility": ["re", "spon", "si", "bil", "i", "ty"],
    "understanding": ["un", "der", "stand", "ing"],
    "entertainment": ["en", "ter", "tain", "ment"],
    "environment": ["en", "vi", "ron", "ment"],
    "development": ["de", "vel", "op", "ment"],
    "relationship": ["re", "la", "tion", "ship"],
    "immediately": ["im", "me", "di", "ate", "ly"],
    "unfortunately": ["un", "for", "tu", "nate", "ly"],
    "automatically": ["au", "to", "mat", "i", "cal", "ly"],
}

# =============================================================================
# ENGLISH MORPHEME DICTIONARY
# =============================================================================

PREFIXES = {
    "un": "not", "re": "again", "pre": "before", "dis": "not/opposite",
    "mis": "wrong", "over": "too much", "under": "too little", "out": "beyond",
    "sub": "under", "super": "above", "anti": "against", "auto": "self",
    "bi": "two", "tri": "three", "multi": "many", "semi": "half",
    "inter": "between", "intra": "within", "extra": "beyond", "ultra": "extreme",
    "non": "not", "co": "together", "counter": "against", "de": "remove",
    "en": "make", "ex": "out of", "fore": "before", "mid": "middle",
    "post": "after", "trans": "across", "pro": "forward",
}

SUFFIXES = {
    "ing": "present participle", "ed": "past tense", "er": "one who/more",
    "est": "most", "ly": "in manner of", "ness": "state of being",
    "ment": "result of", "tion": "act of", "sion": "act of",
    "able": "capable of", "ible": "capable of", "ful": "full of",
    "less": "without", "ous": "having quality of", "ive": "tending to",
    "al": "relating to", "ial": "relating to", "ic": "relating to",
    "ical": "relating to", "ish": "resembling", "like": "similar to",
    "ward": "direction", "wise": "manner", "dom": "state/realm",
    "hood": "state/condition", "ship": "state/skill", "ry": "place/practice",
    "ery": "place/practice", "ity": "quality", "ty": "quality",
}

MORPHEME_WORDS = {
    # Prefix words
    "unhappy": ["un", "happy"], "redo": ["re", "do"], "preview": ["pre", "view"],
    "disagree": ["dis", "agree"], "misunderstand": ["mis", "understand"],
    "overwork": ["over", "work"], "underestimate": ["under", "estimate"],
    "subway": ["sub", "way"], "superhero": ["super", "hero"],
    "antiwar": ["anti", "war"], "automobile": ["auto", "mobile"],
    "bicycle": ["bi", "cycle"], "triangle": ["tri", "angle"],
    "international": ["inter", "national"], "nonfiction": ["non", "fiction"],
    "coworker": ["co", "worker"], "counteract": ["counter", "act"],
    "decode": ["de", "code"], "enable": ["en", "able"], "export": ["ex", "port"],
    "forecast": ["fore", "cast"], "midnight": ["mid", "night"],
    "postwar": ["post", "war"], "transport": ["trans", "port"],
    
    # Suffix words
    "walking": ["walk", "ing"], "walked": ["walk", "ed"], "walker": ["walk", "er"],
    "fastest": ["fast", "est"], "quickly": ["quick", "ly"], "happiness": ["happi", "ness"],
    "movement": ["move", "ment"], "action": ["act", "ion"], "decision": ["decis", "ion"],
    "readable": ["read", "able"], "flexible": ["flex", "ible"], "hopeful": ["hope", "ful"],
    "careless": ["care", "less"], "famous": ["fam", "ous"], "creative": ["creat", "ive"],
    "natural": ["natur", "al"], "special": ["speci", "al"], "magic": ["mag", "ic"],
    "historical": ["histor", "ical"], "childish": ["child", "ish"], "lifelike": ["life", "like"],
    "backward": ["back", "ward"], "clockwise": ["clock", "wise"], "freedom": ["free", "dom"],
    "childhood": ["child", "hood"], "friendship": ["friend", "ship"], "bakery": ["bake", "ry"],
    
    # Prefix + suffix
    "unhappiness": ["un", "happi", "ness"], "disrespectful": ["dis", "respect", "ful"],
    "misunderstanding": ["mis", "understand", "ing"], "unexpectedly": ["un", "expect", "ed", "ly"],
    "international": ["inter", "nation", "al"], "uncomfortable": ["un", "comfort", "able"],
    "disagreement": ["dis", "agree", "ment"], "reconnection": ["re", "connect", "ion"],
}

# =============================================================================
# SIMPLE ENGLISH SENTENCES
# =============================================================================

SENTENCE_TEMPLATES = [
    "The {adj} {noun} {verb}.",
    "A {noun} is {adj}.",
    "{noun} {verb} {adv}.",
    "The {noun} and the {noun} {verb}.",
    "I {verb} the {adj} {noun}.",
    "She {verb} a {noun}.",
    "He {verb} {adv}.",
    "They {verb} in the {noun}.",
    "We {verb} every {noun}.",
    "The {adj} {noun} {verb} the {noun}.",
]

NOUNS = ["cat", "dog", "bird", "fish", "tree", "house", "car", "book", "sun", "moon",
         "child", "man", "woman", "friend", "teacher", "student", "city", "country",
         "water", "food", "music", "movie", "game", "story", "picture", "garden"]

VERBS = ["runs", "walks", "jumps", "plays", "reads", "writes", "sings", "dances",
         "sleeps", "eats", "drinks", "works", "studies", "talks", "listens", "watches"]

ADJECTIVES = ["big", "small", "fast", "slow", "happy", "sad", "good", "bad",
              "new", "old", "hot", "cold", "red", "blue", "green", "white"]

ADVERBS = ["quickly", "slowly", "happily", "sadly", "loudly", "quietly",
           "carefully", "easily", "hard", "well", "often", "always"]


def generate_sentence():
    """Generate a random simple English sentence."""
    template = random.choice(SENTENCE_TEMPLATES)
    sentence = template.format(
        noun=random.choice(NOUNS),
        verb=random.choice(VERBS),
        adj=random.choice(ADJECTIVES),
        adv=random.choice(ADVERBS),
    )
    return sentence


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_syllable_data(num_samples=10000):
    """Generate syllable training data."""
    data = []
    words = list(SYLLABLE_DICT.keys())
    
    for _ in range(num_samples):
        word = random.choice(words)
        syllables = SYLLABLE_DICT[word]
        
        # Create boundary labels (1 = start of syllable)
        boundaries = []
        char_idx = 0
        for syl in syllables:
            for i, c in enumerate(syl):
                boundaries.append(1 if i == 0 else 0)
        
        data.append({
            "word": word,
            "syllables": syllables,
            "syllable_str": "-".join(syllables),
            "boundaries": boundaries,
        })
    
    return data


def generate_morpheme_data(num_samples=5000):
    """Generate morpheme training data."""
    data = []
    words = list(MORPHEME_WORDS.keys())
    
    for _ in range(num_samples):
        word = random.choice(words)
        morphemes = MORPHEME_WORDS[word]
        
        # Classify morpheme types
        types = []
        for m in morphemes:
            if m in PREFIXES:
                types.append("PREFIX")
            elif m in SUFFIXES:
                types.append("SUFFIX")
            else:
                types.append("ROOT")
        
        data.append({
            "word": word,
            "morphemes": morphemes,
            "types": types,
            "morpheme_str": "+".join(morphemes),
        })
    
    return data


def generate_sentence_data(num_samples=20000):
    """Generate sentence training data."""
    data = []
    
    for _ in range(num_samples):
        sentence = generate_sentence()
        words = sentence.replace(".", "").split()
        
        # Get syllables for each word
        word_syllables = []
        for w in words:
            w_lower = w.lower()
            if w_lower in SYLLABLE_DICT:
                word_syllables.append(SYLLABLE_DICT[w_lower])
            else:
                word_syllables.append([w_lower])
        
        data.append({
            "sentence": sentence,
            "words": words,
            "word_syllables": word_syllables,
        })
    
    return data


def save_data():
    """Generate and save all training data."""
    print("=" * 60)
    print("Generating English Basics Training Data")
    print("=" * 60)
    
    # Syllable data
    print("\n📚 Generating syllable data...")
    syllable_data = generate_syllable_data(10000)
    with open(OUTPUT_DIR / "syllables.jsonl", "w") as f:
        for item in syllable_data:
            f.write(json.dumps(item) + "\n")
    print(f"   ✅ Saved {len(syllable_data)} syllable samples")
    
    # Morpheme data
    print("\n📚 Generating morpheme data...")
    morpheme_data = generate_morpheme_data(5000)
    with open(OUTPUT_DIR / "morphemes.jsonl", "w") as f:
        for item in morpheme_data:
            f.write(json.dumps(item) + "\n")
    print(f"   ✅ Saved {len(morpheme_data)} morpheme samples")
    
    # Sentence data
    print("\n📚 Generating sentence data...")
    sentence_data = generate_sentence_data(20000)
    with open(OUTPUT_DIR / "sentences.jsonl", "w") as f:
        for item in sentence_data:
            f.write(json.dumps(item) + "\n")
    print(f"   ✅ Saved {len(sentence_data)} sentence samples")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary:")
    print(f"   Syllable dictionary: {len(SYLLABLE_DICT)} words")
    print(f"   Morpheme dictionary: {len(MORPHEME_WORDS)} words")
    print(f"   Prefixes: {len(PREFIXES)}")
    print(f"   Suffixes: {len(SUFFIXES)}")
    print(f"\n✅ Data saved to {OUTPUT_DIR}/")
    
    # Show samples
    print("\n" + "=" * 60)
    print("📝 Samples:")
    print("=" * 60)
    
    print("\nSyllables:")
    for s in syllable_data[:5]:
        print(f"   {s['word']} → {s['syllable_str']}")
    
    print("\nMorphemes:")
    for m in morpheme_data[:5]:
        print(f"   {m['word']} → {m['morpheme_str']} ({', '.join(m['types'])})")
    
    print("\nSentences:")
    for sent in sentence_data[:5]:
        print(f"   {sent['sentence']}")


if __name__ == "__main__":
    save_data()
