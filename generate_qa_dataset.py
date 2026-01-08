import json
import random
import os

def generate_dataset(output_file, target_size=50000):
    # Word Lists
    m_nouns = ["Hund", "Baum", "Tisch", "Stuhl", "Apfel", "Ball", "Vogel", "Park", "Garten", "Lehrer", "Arzt", "Wald", "Berg", "See", "Fluss", "Schlüssel", "Computer", "Wagen", "Bär", "Löwe"]
    f_nouns = ["Katze", "Blume", "Schule", "Stadt", "Mutter", "Schwester", "Tasche", "Tasse", "Sonne", "Wolke", "Lampe", "Tür", "Zeitung", "Gabel", "Ente", "Maus", "Pflanze", "Insel", "Küche", "Straße"]
    n_nouns = ["Haus", "Auto", "Kind", "Buch", "Mädchen", "Zimmer", "Brot", "Wasser", "Fahrrad", "Fenster", "Pferd", "Schaf", "Licht", "Handy", "Bett", "Kissen", "Glas", "Messer", "Dorf", "Feld"]
    
    all_nouns = m_nouns + f_nouns + n_nouns
    
    names = ["Anna", "Lukas", "Julia", "Max", "Sophie", "Paul", "Emma", "Tom", "Marie", "Jan", "Sarah", "Felix", "Lena", "Tim", "Lisa", "Ben", "Mia", "Noah", "Laura", "Leo"]
    
    colors = ["rot", "blau", "grün", "gelb", "schwarz", "weiß", "grau", "braun", "orange", "lila"]
    adjectives = ["groß", "klein", "neu", "alt", "schön", "hässlich", "schnell", "langsam", "gut", "schlecht", "teuer", "billig", "müde", "glücklich", "traurig", "stark", "schwach", "hell", "dunkel", "warm"]
    
    places = ["im Garten", "in der Schule", "zu Hause", "im Park", "in der Stadt", "im Wald", "im Zimmer", "auf dem Tisch", "unter dem Bett", "vor dem Haus", "hinter dem Baum", "im Büro", "im Kino", "im Restaurant", "am Bahnhof"]
    
    verbs_present = ["läuft", "schläft", "spielt", "liest", "schreibt", "arbeitet", "lernt", "lacht", "weint", "singt", "tanzt", "kocht", "isst", "trinkt", "fährt", "geht"]
    
    times = ["heute", "morgen", "am Montag", "am Dienstag", "am Wochenende", "um acht Uhr", "um zehn Uhr", "im Sommer", "im Winter", "bald", "jetzt", "später"]
    
    categories = {
        "Hund": "ein Tier", "Katze": "ein Haustier", "Vogel": "ein Tier", "Apfel": "eine Frucht", "Brot": "ein Lebensmittel",
        "Baum": "eine Pflanze", "Blume": "eine Pflanze", "Haus": "ein Gebäude", "Auto": "ein Fahrzeug", "Buch": "ein Gegenstand",
        "Tisch": "ein Möbelstück", "Stuhl": "ein Möbelstück", "Sonne": "ein Stern", "Wasser": "ein Getränk", "Handy": "ein Gerät"
    }

    dataset = []
    seen_questions = set()

    def add_pair(q, a):
        if q not in seen_questions:
            if 10 <= len(a) <= 80:
                dataset.append({"question": q, "answer": a})
                seen_questions.add(q)

    # 1. WAS (What)
    for noun, cat in categories.items():
        add_pair(f"Was ist ein {noun}?", f"Ein {noun} ist {cat}.")
        add_pair(f"Was ist das?", f"Das ist ein {noun}.")
        
    for noun in all_nouns:
        adj = random.choice(adjectives)
        color = random.choice(colors)
        add_pair(f"Was ist {adj}?", f"Das {noun} ist {adj}.")
        add_pair(f"Welche Farbe hat der {noun}?", f"Der {noun} ist {color}.") if noun in m_nouns else None
        add_pair(f"Welche Farbe hat die {noun}?", f"Die {noun} ist {color}.") if noun in f_nouns else None
        add_pair(f"Welche Farbe hat das {noun}?", f"Das {noun} ist {color}.") if noun in n_nouns else None

    # 2. WER (Who)
    for name in names:
        role = random.choice(["mein Freund", "meine Schwester", "der Lehrer", "ein Kind", "der Nachbar", "die Ärztin"])
        add_pair(f"Wer ist {name}?", f"{name} ist {role}.")
        verb = random.choice(verbs_present)
        add_pair(f"Wer {verb}?", f"{name} {verb}.")

    # 3. WO (Where)
    for noun in all_nouns:
        loc = random.choice(places)
        prefix = "Der" if noun in m_nouns else "Die" if noun in f_nouns else "Das"
        add_pair(f"Wo ist {prefix.lower()} {noun}?", f"{prefix} {noun} ist {loc}.")
        
    for name in names:
        loc = random.choice(places)
        add_pair(f"Wo ist {name}?", f"{name} ist {loc}.")

    # 4. WANN (When)
    events = ["die Schule", "der Film", "das Spiel", "das Essen", "die Party", "der Kurs", "das Treffen"]
    for event in events:
        time = random.choice(times)
        add_pair(f"Wann beginnt {event}?", f"{event.capitalize()} beginnt {time}.")
        add_pair(f"Wann ist {event}?", f"{event.capitalize()} ist {time}.")

    # 5. WARUM (Why)
    reasons = [
        ("Warum bist du traurig?", "Ich habe mein Buch verloren."),
        ("Warum lachst du?", "Der Witz war sehr lustig."),
        ("Warum ist er müde?", "Er hat viel gearbeitet."),
        ("Warum regnet es?", "Es sind viele Wolken am Himmel."),
        ("Warum lernst du?", "Ich habe morgen eine Prüfung."),
        ("Warum gehst du?", "Ich muss nach Hause gehen.")
    ]
    for q, a in reasons:
        add_pair(q, a)

    # 6. WIE (How)
    for name in names:
        age = random.randint(5, 80)
        add_pair(f"Wie alt ist {name}?", f"{name} ist {age} Jahre alt.")
        add_pair(f"Wie geht es {name}?", f"Es geht {name} sehr gut.")
    
    add_pair("Wie ist das Wetter?", "Das Wetter ist heute sehr schön.")
    add_pair("Wie viel Uhr ist es?", "Es ist jetzt genau zwei Uhr.")

    # 7. JA/NEIN (Yes/No)
    for noun in all_nouns:
        prefix = "den" if noun in m_nouns else "die" if noun in f_nouns else "das"
        add_pair(f"Hast du {prefix} {noun}?", f"Ja, ich habe {prefix} {noun}.")
        add_pair(f"Magst du {noun}?", f"Nein, ich mag keine {noun}.")
        
    for verb in verbs_present:
        add_pair(f"{verb.capitalize()} du?", f"Ja, ich {verb} gerne.")

    # 8. Combinatorial Expansion
    while len(dataset) < target_size:
        # Generate more by mixing
        n = random.choice(all_nouns)
        adj = random.choice(adjectives)
        loc = random.choice(places)
        name = random.choice(names)
        v = random.choice(verbs_present)
        t = random.choice(times)
        
        # Random patterns
        choice = random.randint(1, 10)
        if choice == 1:
            add_pair(f"Ist das {n} {adj}?", f"Ja, das {n} ist sehr {adj}.")
        elif choice == 2:
            add_pair(f"Wer {v} {loc}?", f"{name} {v} dort.")
        elif choice == 3:
            add_pair(f"Was macht {name} {t}?", f"{name} {v} {t}.")
        elif choice == 4:
            add_pair(f"Wo {v} der {n}?", f"Der {n} {v} {loc}.")
        elif choice == 5:
            add_pair(f"Wann {v} {name}?", f"{name} {v} {t}.")
        elif choice == 6:
            prefix = "einen" if n in m_nouns else "eine" if n in f_nouns else "ein"
            add_pair(f"Siehst du {prefix} {n}?", f"Ja, ich sehe {prefix} {n} {loc}.")
        elif choice == 7:
            add_pair(f"Warum ist der {n} {adj}?", f"Weil er schon sehr alt ist.")
        elif choice == 8:
            add_pair(f"Wie findet {name} das {n}?", f"{name} findet das {n} {adj}.")
        elif choice == 9:
            add_pair(f"Welches {n} ist {adj}?", f"Das {n} {loc} ist {adj}.")
        elif choice == 10:
            add_pair(f"Hast du {t} Zeit?", f"Ja, ich habe {t} viel Zeit.")

        if len(seen_questions) > target_size * 2: # Prevent infinite loop if we run out of unique pairs
            break

    random.shuffle(dataset)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset[:target_size], f, ensure_ascii=False, indent=4)
    
    return len(dataset)

if __name__ == "__main__":
    count = generate_dataset("german_qa_dataset.json", 25000)
    print(f"Generated {count} Q&A pairs in german_qa_dataset.json")
