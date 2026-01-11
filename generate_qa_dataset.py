import json
import random
import os
import itertools
from collections import Counter

# --- GLOBAL ASSETS: MASSIVE CONSOLIDATED VOCABULARY ---
VOCAB_RAW = [
    # Masculine (der) - g:0
    ("Hund", 0, 'm', True, "ein Tier"), ("Baum", 0, 'l', False, "eine Pflanze"),
    ("Tisch", 0, 'm', False, "ein Möbelstück"), ("Stuhl", 0, 'm', False, "ein Möbelstück"),
    ("Apfel", 0, 's', False, "eine Frucht"), ("Ball", 0, 's', False, "ein Spielzeug"),
    ("Vogel", 0, 's', True, "ein Tier"), ("Park", 0, 'h', False, "ein Ort"),
    ("Garten", 0, 'h', False, "ein Ort"), ("Wald", 0, 'h', False, "ein Ort"),
    ("Berg", 0, 'h', False, "ein Berg"), ("See", 0, 'h', False, "ein Gewässer"),
    ("Fluss", 0, 'h', False, "ein Gewässer"), ("Schlüssel", 0, 's', False, "ein Gegenstand"),
    ("Computer", 0, 'm', False, "ein Gerät"), ("Wagen", 0, 'l', False, "ein Fahrzeug"),
    ("Kühlschrank", 0, 'l', False, "ein Haushaltsgerät"), ("Löffel", 0, 's', False, "ein Besteck"),
    ("Schrank", 0, 'l', False, "ein Möbelstück"), ("Bus", 0, 'h', False, "ein Fahrzeug"),
    ("Zug", 0, 'h', False, "ein Fahrzeug"), ("Hut", 0, 's', False, "eine Kopfbedeckung"),
    ("Mantel", 0, 's', False, "ein Kleidungsstück"), ("Kuchen", 0, 's', False, "ein Gebäck"),
    ("Teller", 0, 's', False, "ein Geschirr"), ("Stift", 0, 's', False, "ein Schreibgerät"),
    ("Hammer", 0, 's', False, "ein Werkzeug"), ("Topf", 0, 's', False, "ein Kochgeschirr"),
    ("Besen", 0, 'm', False, "ein Reinigungsgerät"), ("Teppich", 0, 'm', False, "ein Einrichtungsgegenstand"),
    ("Sessel", 0, 'm', False, "ein Möbelstück"), ("Spiegel", 0, 'm', False, "ein Gegenstand"),
    ("Kamm", 0, 's', False, "ein Pflegeprodukt"), ("Gürtel", 0, 's', False, "ein Accessoire"),
    ("Regenschirm", 0, 's', False, "ein Wetterschutz"), ("Rucksack", 0, 's', False, "eine Tasche"),
    ("Koffer", 0, 'm', False, "ein Reisegepäck"), ("Becher", 0, 's', False, "ein Gefäß"),
    ("Schuh", 0, 's', False, "eine Fußbekleidung"), ("Schal", 0, 's', False, "ein Accessoire"),
    ("Bär", 0, 'l', True, "ein Wildtier"), ("Löwe", 0, 'l', True, "ein Raubtier"),
    ("Wolf", 0, 'l', True, "ein Raubtier"), ("Elefant", 0, 'h', True, "ein Dickhäuter"),
    ("Fisch", 0, 's', True, "ein Wassertier"), ("Hahn", 0, 's', True, "ein Nutztier"),
    ("Tiger", 0, 'l', True, "ein Raubtier"), ("Hirsch", 0, 'l', True, "ein Wildtier"),
    ("Hase", 0, 's', True, "ein Wildtier"), ("Igel", 0, 's', True, "ein Wildtier"),
    ("Esel", 0, 'l', True, "ein Nutztier"), ("Affe", 0, 'm', True, "ein Primat"),
    ("Papagei", 0, 's', True, "ein Vogel"), ("Wal", 0, 'h', True, "ein Wassertier"),
    ("Delfin", 0, 'l', True, "ein Wassertier"), ("Lachs", 0, 's', True, "ein Fisch"),
    ("Saft", 0, 's', False, "ein Getränk"), ("Kaffee", 0, 's', False, "ein Getränk"),
    ("Wein", 0, 's', False, "ein Getränk"), ("Tee", 0, 's', False, "ein Getränk"),
    ("Käse", 0, 's', False, "ein Lebensmittel"), ("Salat", 0, 's', False, "ein Gericht"),
    ("Honig", 0, 's', False, "ein Lebensmittel"), ("Zucker", 0, 's', False, "ein Lebensmittel"),
    ("Reis", 0, 's', False, "ein Lebensmittel"), ("Pfeffer", 0, 's', False, "ein Gewürz"),
    ("Stein", 0, 's', False, "ein Mineral"), ("Ring", 0, 's', False, "ein Schmuckstück"),
    ("Turm", 0, 'h', False, "ein Bauwerk"), ("Dom", 0, 'h', False, "ein Gebäude"),
    ("Palast", 0, 'h', False, "ein Gebäude"), ("Monitor", 0, 'm', False, "ein Gerät"),
    ("Drucker", 0, 'm', False, "ein Gerät"), ("Laptop", 0, 'm', False, "ein Gerät"),
    ("Bürostuhl", 0, 'm', False, "ein Möbelstück"), ("Schreibtisch", 0, 'm', False, "ein Möbelstück"),
    ("Zaun", 0, 'l', False, "eine Begrenzung"), ("Flughafen", 0, 'h', False, "ein Ort"),
    ("Bahnhof", 0, 'h', False, "ein Ort"), ("Hafen", 0, 'h', False, "ein Ort"),
    ("Vater", 0, 'm', True, "eine Person"), ("Sohn", 0, 'm', True, "eine Person"),
    ("Bruder", 0, 'm', True, "eine Person"), ("Lehrer", 0, 'm', True, "eine Person"),
    ("Arzt", 0, 'm', True, "eine Person"), ("Koch", 0, 'm', True, "eine Person"),
    ("Polizist", 0, 'm', True, "eine Person"), ("König", 0, 'm', True, "eine Person"),
    ("Soldat", 0, 'm', True, "eine Person"), ("Pilot", 0, 'm', True, "eine Person"),
    ("Anwalt", 0, 'm', True, "eine Person"), ("Held", 0, 'm', True, "eine Person"),
    
    # Feminine (die) - g:1
    ("Katze", 1, 's', True, "ein Haustier"), ("Blume", 1, 's', False, "eine Pflanze"),
    ("Schule", 1, 'h', False, "eine Einrichtung"), ("Stadt", 1, 'h', False, "ein Ort"),
    ("Mutter", 1, 'm', True, "eine Person"), ("Schwester", 1, 'm', True, "eine Person"),
    ("Tasche", 1, 's', False, "ein Behältnis"), ("Tasse", 1, 's', False, "ein Gefäß"),
    ("Sonne", 1, 'h', False, "ein Stern"), ("Wolke", 1, 'h', False, "ein Wetterphänomen"),
    ("Lampe", 1, 's', False, "eine Lichtquelle"), ("Tür", 1, 'm', False, "ein Bauelement"),
    ("Zeitung", 1, 's', False, "ein Medium"), ("Gabel", 1, 's', False, "ein Besteck"),
    ("Ente", 1, 's', True, "ein Vogel"), ("Maus", 1, 's', True, "ein Nagetier"),
    ("Insel", 1, 'h', False, "eine Landmasse"), ("Küche", 1, 'm', False, "ein Raum"),
    ("Straße", 1, 'h', False, "ein Verkehrsweg"), ("Banane", 1, 's', False, "eine Frucht"),
    ("Gitarre", 1, 'm', False, "ein Instrument"), ("Uhr", 1, 's', False, "ein Zeitmesser"),
    ("Jacke", 1, 's', False, "ein Kleidungsstück"), ("Pfanne", 1, 's', False, "ein Kochgerät"),
    ("Ärztin", 1, 'm', True, "eine Person"), ("Lehrerin", 1, 'm', True, "eine Person"),
    ("Oma", 1, 'm', True, "eine Person"), ("Tante", 1, 'm', True, "eine Person"),
    ("Kuh", 1, 'l', True, "ein Nutztier"), ("Ziege", 1, 'm', True, "ein Nutztier"),
    ("Milch", 1, 's', False, "ein Getränk"), ("Wurst", 1, 's', False, "ein Lebensmittel"),
    ("Suppe", 1, 's', False, "eine Speise"), ("Pizza", 1, 's', False, "eine Speise"),
    ("Zitrone", 1, 's', False, "eine Frucht"), ("Kirche", 1, 'h', False, "ein religiöses Gebäude"),
    ("Bank", 1, 'm', False, "eine Institution"), ("Post", 1, 'm', False, "eine Institution"),
    ("Wiese", 1, 'h', False, "eine Fläche"), ("Brücke", 1, 'h', False, "ein Bauwerk"),
    ("Polizei", 1, 'h', False, "eine Behörde"), ("Burg", 1, 'h', False, "ein Bauwerk"),
    ("Villa", 1, 'h', False, "ein Gebäude"), ("Hütte", 1, 'm', False, "ein Gebäude"),
    ("Frau", 1, 'm', True, "eine Person"), ("Ärztin", 1, 'm', True, "eine Person"),
    ("Rose", 1, 's', False, "eine Blume"), ("Welt", 1, 'h', False, "ein Ort"),
    ("Birne", 1, 's', False, "eine Frucht"), ("Ente", 1, 's', True, "ein Vogel"),
    ("Gans", 1, 'm', True, "ein Nutztier"), ("Biene", 1, 's', True, "ein Insekt"),
    
    # Neuter (das) - g:2
    ("Haus", 2, 'h', False, "ein Gebäude"), ("Auto", 2, 'h', False, "ein Fahrzeug"),
    ("Kind", 2, 'm', True, "eine Person"), ("Buch", 2, 's', False, "ein Gegenstand"),
    ("Mädchen", 2, 'm', True, "eine Person"), ("Zimmer", 2, 'm', False, "ein Raum"),
    ("Brot", 2, 's', False, "ein Lebensmittel"), ("Wasser", 2, 'h', False, "ein Getränk"),
    ("Fahrrad", 2, 'l', False, "ein Fahrzeug"), ("Fenster", 2, 'm', False, "ein Bauelement"),
    ("Pferd", 2, 'l', True, "ein Tier"), ("Schaf", 2, 'l', True, "ein Nutztier"),
    ("Handy", 2, 's', False, "ein Gerät"), ("Bett", 2, 'l', False, "ein Möbelstück"),
    ("Glas", 2, 's', False, "ein Gefäß"), ("Messer", 2, 's', False, "ein Besteck"),
    ("Dorf", 2, 'h', False, "ein Ort"), ("Flugzeug", 2, 'h', False, "ein Fahrzeug"),
    ("Schiff", 2, 'h', False, "ein Fahrzeug"), ("Kleid", 2, 's', False, "ein Kleidungsstück"),
    ("Ei", 2, 's', False, "ein Lebensmittel"), ("Gemüse", 2, 's', False, "ein Lebensmittel"),
    ("Kalb", 2, 'm', True, "ein Tier"), ("Baby", 2, 's', True, "eine Person"),
    ("Schwein", 2, 'l', True, "ein Nutztier"), ("Kino", 2, 'h', False, "ein Ort"),
    ("Theater", 2, 'h', False, "ein Ort"), ("Museum", 2, 'h', False, "ein Ort"),
    ("Hotel", 2, 'h', False, "ein Ort"), ("Bild", 2, 's', False, "ein Kunstwerk"),
    ("Spiel", 2, 's', False, "eine Tätigkeit"), ("Heft", 2, 's', False, "ein Schreibmedium"),
    ("Zelt", 2, 'l', False, "eine Unterkunft"), ("Boot", 2, 'l', False, "ein Wasserfahrzeug"),
    ("Gold", 2, 's', False, "ein Edelmetall"), ("Silber", 2, 's', False, "ein Edelmetall"),
    ("Tablet", 2, 's', False, "ein Gerät"), ("Klavier", 2, 'm', False, "ein Musikinstrument"),
    ("Stadion", 2, 'h', False, "ein Bauwerk"), ("Amt", 2, 'm', False, "eine Behörde"),
    ("Herz", 2, 's', True, "ein Organ"), ("Meer", 2, 'h', False, "ein Gewässer"),
    ("Tal", 2, 'h', False, "ein Naturmerkmal"), ("Gras", 2, 's', False, "eine Pflanze"),
    ("Geschenk", 2, 's', False, "ein Gegenstand"), ("Lied", 2, 's', False, "ein Medium"),
    ("Schloss", 2, 'h', False, "ein Gebäude"), ("Wort", 2, 's', False, "ein Zeichen"),
    ("Tier", 2, 's', True, "ein Lebewesen"), ("Fenster", 2, 'm', False, "ein Bauelement"),
]

# Extended vocabulary with real German nouns
VOCAB_RAW.extend([
    # More masculine nouns
    ("Apfel", 0, 's', False, "eine Frucht"), ("Kuchen", 0, 's', False, "ein Gebäck"),
    ("Schrank", 0, 'l', False, "ein Möbelstück"), ("Koffer", 0, 'm', False, "ein Behälter"),
    ("Schlüssel", 0, 's', False, "ein Werkzeug"), ("Spiegel", 0, 'm', False, "ein Gegenstand"),
    ("Teppich", 0, 'l', False, "ein Textil"), ("Regenschirm", 0, 'm', False, "ein Accessoire"),
    ("Rucksack", 0, 'm', False, "eine Tasche"), ("Schreibtisch", 0, 'l', False, "ein Möbelstück"),
    ("Fernseher", 0, 'l', False, "ein Gerät"), ("Kühlschrank", 0, 'l', False, "ein Gerät"),
    ("Staubsauger", 0, 'm', False, "ein Gerät"), ("Wecker", 0, 's', False, "ein Gerät"),
    ("Kalender", 0, 's', False, "ein Gegenstand"), ("Bleistift", 0, 's', False, "ein Schreibgerät"),
    ("Kugelschreiber", 0, 's', False, "ein Schreibgerät"), ("Ordner", 0, 'm', False, "ein Behälter"),
    ("Brief", 0, 's', False, "ein Medium"), ("Umschlag", 0, 's', False, "ein Behälter"),
    ("Schuh", 0, 's', False, "ein Kleidungsstück"), ("Hut", 0, 's', False, "eine Kopfbedeckung"),
    ("Gürtel", 0, 's', False, "ein Accessoire"), ("Handschuh", 0, 's', False, "ein Kleidungsstück"),
    ("Schal", 0, 's', False, "ein Kleidungsstück"), ("Mantel", 0, 'm', False, "ein Kleidungsstück"),
    ("Anzug", 0, 'm', False, "ein Kleidungsstück"), ("Pullover", 0, 'm', False, "ein Kleidungsstück"),
    ("Käse", 0, 's', False, "ein Lebensmittel"), ("Schinken", 0, 's', False, "ein Lebensmittel"),
    ("Salat", 0, 's', False, "ein Lebensmittel"), ("Pilz", 0, 's', False, "ein Lebensmittel"),
    ("Fisch", 0, 's', False, "ein Lebensmittel"), ("Braten", 0, 'm', False, "ein Gericht"),
    ("Löwe", 0, 'l', True, "ein Raubtier"), ("Bär", 0, 'l', True, "ein Wildtier"),
    ("Wolf", 0, 'm', True, "ein Raubtier"), ("Fuchs", 0, 'm', True, "ein Wildtier"),
    ("Adler", 0, 'm', True, "ein Vogel"), ("Spatz", 0, 's', True, "ein Vogel"),
    ("Schmetterling", 0, 's', True, "ein Insekt"), ("Käfer", 0, 's', True, "ein Insekt"),
    ("Frosch", 0, 's', True, "ein Tier"), ("Fisch", 0, 's', True, "ein Wassertier"),
    ("Berg", 0, 'h', False, "ein Naturmerkmal"), ("Fluss", 0, 'h', False, "ein Gewässer"),
    ("See", 0, 'h', False, "ein Gewässer"), ("Strand", 0, 'h', False, "ein Ort"),
    ("Wald", 0, 'h', False, "ein Ort"), ("Park", 0, 'h', False, "ein Ort"),
    ("Markt", 0, 'h', False, "ein Ort"), ("Platz", 0, 'h', False, "ein Ort"),
    ("Weg", 0, 'h', False, "ein Verkehrsweg"), ("Pfad", 0, 'm', False, "ein Verkehrsweg"),
    # More feminine nouns
    ("Tasche", 1, 'm', False, "ein Behälter"), ("Flasche", 1, 's', False, "ein Behälter"),
    ("Schüssel", 1, 's', False, "ein Geschirr"), ("Tasse", 1, 's', False, "ein Geschirr"),
    ("Vase", 1, 's', False, "ein Gegenstand"), ("Kerze", 1, 's', False, "eine Lichtquelle"),
    ("Decke", 1, 'm', False, "ein Textil"), ("Matratze", 1, 'l', False, "ein Möbelstück"),
    ("Gardine", 1, 'm', False, "ein Textil"), ("Tapete", 1, 'm', False, "ein Material"),
    ("Treppe", 1, 'm', False, "ein Bauelement"), ("Wand", 1, 'h', False, "ein Bauelement"),
    ("Dusche", 1, 'm', False, "ein Sanitärobjekt"), ("Badewanne", 1, 'l', False, "ein Sanitärobjekt"),
    ("Waschmaschine", 1, 'l', False, "ein Gerät"), ("Spülmaschine", 1, 'l', False, "ein Gerät"),
    ("Heizung", 1, 'm', False, "ein Gerät"), ("Klimaanlage", 1, 'l', False, "ein Gerät"),
    ("Socke", 1, 's', False, "ein Kleidungsstück"), ("Hose", 1, 'm', False, "ein Kleidungsstück"),
    ("Bluse", 1, 's', False, "ein Kleidungsstück"), ("Mütze", 1, 's', False, "eine Kopfbedeckung"),
    ("Brille", 1, 's', False, "eine Sehhilfe"), ("Kette", 1, 's', False, "ein Schmuckstück"),
    ("Orange", 1, 's', False, "eine Frucht"), ("Traube", 1, 's', False, "eine Frucht"),
    ("Erdbeere", 1, 's', False, "eine Frucht"), ("Kirsche", 1, 's', False, "eine Frucht"),
    ("Kartoffel", 1, 's', False, "ein Gemüse"), ("Tomate", 1, 's', False, "ein Gemüse"),
    ("Gurke", 1, 's', False, "ein Gemüse"), ("Zwiebel", 1, 's', False, "ein Gemüse"),
    ("Möhre", 1, 's', False, "ein Gemüse"), ("Bohne", 1, 's', False, "ein Gemüse"),
    ("Nudel", 1, 's', False, "ein Lebensmittel"), ("Torte", 1, 's', False, "ein Gebäck"),
    ("Spinne", 1, 's', True, "ein Tier"), ("Ameise", 1, 's', True, "ein Insekt"),
    ("Taube", 1, 's', True, "ein Vogel"), ("Möwe", 1, 's', True, "ein Vogel"),
    ("Eule", 1, 's', True, "ein Vogel"), ("Schwalbe", 1, 's', True, "ein Vogel"),
    ("Schildkröte", 1, 'm', True, "ein Reptil"), ("Schlange", 1, 'm', True, "ein Reptil"),
    ("Wolke", 1, 'h', False, "ein Naturphänomen"), ("Sonne", 1, 'h', False, "ein Himmelskörper"),
    ("Nacht", 1, 'h', False, "eine Tageszeit"), ("Woche", 1, 'm', False, "ein Zeitraum"),
    # More neuter nouns
    ("Sofa", 2, 'l', False, "ein Möbelstück"), ("Regal", 2, 'l', False, "ein Möbelstück"),
    ("Kissen", 2, 's', False, "ein Textil"), ("Handtuch", 2, 's', False, "ein Textil"),
    ("Besteck", 2, 's', False, "ein Gegenstand"), ("Geschirr", 2, 's', False, "ein Gegenstand"),
    ("Radio", 2, 'm', False, "ein Gerät"), ("Telefon", 2, 'm', False, "ein Gerät"),
    ("Mikrofon", 2, 's', False, "ein Gerät"), ("Keyboard", 2, 's', False, "ein Gerät"),
    ("Hemd", 2, 's', False, "ein Kleidungsstück"), ("T-Shirt", 2, 's', False, "ein Kleidungsstück"),
    ("Armband", 2, 's', False, "ein Schmuckstück"), ("Parfum", 2, 's', False, "ein Kosmetikprodukt"),
    ("Obst", 2, 's', False, "ein Lebensmittel"), ("Bier", 2, 's', False, "ein Getränk"),
    ("Mehl", 2, 's', False, "ein Lebensmittel"), ("Salz", 2, 's', False, "ein Gewürz"),
    ("Öl", 2, 's', False, "ein Lebensmittel"), ("Gewürz", 2, 's', False, "ein Lebensmittel"),
    ("Insekt", 2, 's', True, "ein Tier"), ("Eichhörnchen", 2, 's', True, "ein Nagetier"),
    ("Kaninchen", 2, 's', True, "ein Nagetier"), ("Meerschweinchen", 2, 's', True, "ein Nagetier"),
    ("Krokodil", 2, 'l', True, "ein Reptil"), ("Nashorn", 2, 'l', True, "ein Wildtier"),
    ("Zebra", 2, 'l', True, "ein Wildtier"), ("Kamel", 2, 'l', True, "ein Tier"),
    ("Wetter", 2, 'h', False, "ein Naturphänomen"), ("Gewitter", 2, 'h', False, "ein Naturphänomen"),
    ("Gebirge", 2, 'h', False, "ein Naturmerkmal"), ("Ufer", 2, 'h', False, "ein Ort"),
    ("Denkmal", 2, 'h', False, "ein Bauwerk"), ("Rathaus", 2, 'h', False, "ein Gebäude"),
    ("Krankenhaus", 2, 'h', False, "ein Gebäude"), ("Schwimmbad", 2, 'h', False, "ein Ort"),
])

NOUNS_DB = {item[0]: {'g': item[1], 's': item[2], 'l': item[3]} for item in VOCAB_RAW}
CAT_MAP = {item[0]: item[4] for item in VOCAB_RAW}

NAMES = ["Anna", "Lukas", "Julia", "Max", "Sophie", "Paul", "Emma", "Tom", "Marie", "Jan", "Sarah", "Felix", "Lena", "Tim", "Lisa", "Ben", "Mia", "Noah", "Laura", "Leo", "Jonas", "Hanna", "Erik", "Lara", "Finn", "Markus", "Sandra", "Klaus", "Monika", "Uwe", "Sabine", "Stefan", "Petra", "Andreas", "Karin", "Thomas", "Nicole", "Christian", "Bärbel", "Dieter", "Wolfgang", "Renate", "Jürgen", "Helga", "Werner", "Ursula", "Manfred", "Ingrid", "Günter", "Hans", "Gisela", "Helmut", "Erika", "Bernd", "Hildegard", "Peter", "Brigitte", "Karl", "Marianne", "Joachim", "Christa", "Siegfried", "Edith", "Herbert", "Anneliese", "Gottfried", "Rosemarie", "Heinrich", "Waltraud", "Karl-Heinz", "Gerhard", "Angelika", "Günther", "Hans-Joachim", "Christel", "Bernhard", "Helene", "Ewald", "Lydia", "Friedrich", "Dorothea", "Wilhelm", "Eleonore", "Otto", "Martha", "Emil", "Clara", "Arthur", "Johanna", "Ludwig", "Berta", "Theodor", "Agnes", "Hugo", "Elisabeth", "Maria", "Rudolf", "Therese", "Hermann", "Frieda", "Alfred", "Julius", "Louise", "Oskar", "Mathilde", "Robert", "Emilie", "Georg", "Paula", "Richard", "Ida", "Erich", "Herta", "Walther", "Gertrud", "Pauline", "Wilhelmine", "Margarete", "Kurt", "Gerda", "Heinz", "Inge", "Walter", "Hannelore", "Rolf", "Gabriele", "Friedhelm", "Heidrun", "Siegmar", "Elfriede", "Adolf", "Urs", "Beatrix", "Claus", "Doris", "Egbert", "Felicitas", "Gernot", "Hilde", "Ingo", "Jutta", "Kunibert", "Liselotte", "Meinrad", "Nora", "Ottmar", "Pia", "Quirin", "Reginhard", "Saskia", "Tilman", "Uta", "Volker", "Wanda", "Xaver", "Yvonne", "Zeno", "Armin", "Beate", "Carsten", "Dagmar", "Eckhard", "Frauke", "Gregor", "Heike", "Immanuel", "Juliane", "Konrad", "Leonore", "Moritz", "Nadja", "Oliver", "Patrizia", "Quentin", "Rainer", "Sibylle", "Tobias", "Ulrike", "Valentin", "Wibke", "Xenia", "Yannick", "Zita", "Albert", "Berta", "Clemens", "Dora", "Emil", "Frieda", "Gustav", "Herta", "Isidor", "Josefa", "Karl", "Lotte", "Moritz", "Nina", "Oskar", "Paula", "Richard", "Selma", "Theodor", "Ulla", "Viktor", "Wanda", "Xaver", "Yara", "Zeno", "Anton", "Berti", "Carla", "Doris", "Egon", "Fanny", "Gerd", "Hilda", "Ilse", "Jakob", "Käthe", "Lenz", "Marga", "Nils", "Olga", "Poldi", "Quast", "Resi", "Susi", "Toni", "Udo", "Vroni", "Willi", "Xidi", "Yoshi", "Ziska", "Aron", "Beatrice", "Cora", "Dante", "Elsa", "Falco", "Gina", "Hugo", "Irma", "Jago", "Kora", "Lando", "Mina", "Nero", "Ora", "Paco", "Quina", "Ria", "Sven", "Tessa", "Uriel", "Vera", "Wanja", "Xena", "Yago", "Zora", "Albrecht", "Brunhilde", "Caspar", "Dagobert", "Edmund", "Florenz", "Gisbert", "Hadubrand", "Irmgard", "Jost", "Kunigunde", "Lothar", "Mechthild", "Norbert", "Oswald", "Pankraz", "Reinhold", "Sieglinde", "Traugott", "Ulf", "Volkmar", "Waldemar", "Xaverius", "Yorick", "Zygmunt", "Adalbert", "Bastian", "Cornelia", "Dietmar", "Evelyn", "Friederike", "Gudrun", "Hubert", "Ines", "Jochen", "Karin", "Ludger", "Maren", "Nadine", "Olaf", "Philipp", "Rüdiger", "Silke", "Torsten", "Ute", "Verena", "Winfried", "Yvonne", "Zita", "Adrian", "Bianca", "Claudio", "Daniela", "Enrico", "Fabienne", "Giuseppe", "Hanna", "Igor", "Jasmin", "Kevin", "Larissa", "Marco", "Natalia", "Orlando", "Paola", "Ricardo", "Sonia", "Tatjana", "Umberto", "Valeria", "William", "Xenia", "Yann", "Ziva", "Achim", "Beatrix", "Cilly", "Dankmar", "Eberhard", "Fridolin", "Gertrud", "Hardy", "Ilona", "Justus", "Kuno", "Ludmilla", "Meinhard", "Niklas", "Ottilie", "Priska", "Quirinus", "Roderich", "Senta", "Tristan", "Udo", "Valeska", "Wilfried", "Xaver", "Yolanda", "Zita"]

# Extended list of real German names
NAMES.extend([
    "Amelie", "Antonia", "Barbara", "Bettina", "Carina", "Caroline", "Charlotte", "Christina",
    "Diana", "Elena", "Elisa", "Elisabeth", "Emilia", "Eva", "Franziska", "Greta",
    "Helena", "Ida", "Isabel", "Jana", "Johanna", "Katharina", "Klara", "Leonie",
    "Lina", "Luisa", "Magdalena", "Marlene", "Martina", "Melanie", "Mila", "Miriam",
    "Nathalie", "Paula", "Rebecca", "Rosa", "Sabrina", "Sandra", "Simone", "Stefanie",
    "Tanja", "Teresa", "Vanessa", "Viktoria", "Alexandra", "Alina", "Andrea", "Anita",
    "Alexander", "Andreas", "Anton", "Benjamin", "Christoph", "Daniel", "David", "Dominik",
    "Elias", "Fabian", "Florian", "Georg", "Hannes", "Jakob", "Jan", "Johannes",
    "Jonathan", "Julian", "Konstantin", "Lars", "Lennart", "Leon", "Luca", "Luis",
    "Lutz", "Manuel", "Marcel", "Martin", "Matthias", "Michael", "Nico", "Patrick",
    "Rafael", "Robert", "Samuel", "Sebastian", "Simon", "Stefan", "Sven", "Thomas",
    "Tilman", "Uwe", "Vincent", "Wolfgang", "Achmed", "Ahmed", "Ali", "Fatima",
    "Hassan", "Leila", "Mehmet", "Yusuf", "Ayse", "Elif", "Emir", "Amir",
    "Selin", "Deniz", "Cem", "Baris", "Kemal", "Sibel", "Zeynep", "Murat",
    "Birgit", "Brunhild", "Dagmar", "Elfriede", "Gerda", "Gisela", "Gudrun", "Hedwig",
    "Helene", "Irmgard", "Karla", "Lieselotte", "Margot", "Marlies", "Renate", "Ruth",
    "Sigrid", "Traute", "Waltraud", "Alfred", "Artur", "Egon", "Erwin", "Franz",
    "Fritz", "Günther", "Harald", "Heinz", "Horst", "Klaus", "Manfred", "Norbert",
    "Rainer", "Reinhard", "Siegfried", "Willy", "Alois", "Benedikt", "Cornelius", "Dietrich",
    "Engelbert", "Ferdinand", "Gottlieb", "Heinrich", "Ignaz", "Josef", "Leopold", "Ludwig",
    "Maximilian", "Nikolaus", "Oswald", "Philipp", "Rupert", "Severin", "Ulrich", "Wendelin",
    "Xaver", "Zacharias", "Adelheid", "Bertha", "Cordula", "Dorothee", "Ernestine", "Friederike",
    "Genoveva", "Henriette", "Ilse", "Josephine", "Kunigunde", "Ludmilla", "Mechthild", "Notburga",
    "Ottilie", "Pauline", "Rosalie", "Sibylle", "Theresia", "Ursula", "Veronika", "Wilhelmine"
])

ADJ_COLORS = {
    "rot": "rote", "blau": "blaue", "grün": "grüne", "gelb": "gelbe", "schwarz": "schwarze", 
    "weiß": "weiße", "grau": "graue", "braun": "braune", "hellgrau": "hellgraue", 
    "dunkelblau": "dunkelblaue", "weinrot": "weinrote", "marineblau": "marineblaue", 
    "kupfer": "kupferne", "bronze": "bronzene", "anthrazit": "anthrazitfarbene", 
    "ocker": "ockerfarbene", "gold": "goldene", "silber": "silberne", "pastell": "pastellfarbene"
}
INDECL_COLORS = ["orange", "lila", "rosa", "beige", "türkis", "creme", "oliv", "violett", "neon", "indigo", "khaki"]

ADJ_P = ["fleißig", "müde", "glücklich", "traurig", "stark", "schwach", "gut", "schlecht", "lustig", "klug", "jung", "alt", "aktiv", "ruhig", "krank", "gesund", "nett", "höflich", "stolz", "geduldig", "ehrlich", "mutig", "faul", "reich", "arm", "zufrieden", "nervös", "wild", "sanft", "hübsch", "hässlich", "dick", "dünn", "schnell", "langsam", "klug", "dumm", "fit", "müde", "schlau", "naiv", "bescheiden", "froh", "wütend", "wach", "ernst", "offen", "verschlossen", "satt", "durstig", "gelassen", "hektisch", "ungeduldig", "unhöflich", "vorsichtig", "leichtsinnig", "treu", "pünchtlich", "ordentlich", "unordentlich", "warm", "kalt", "heiß", "eisig", "trocken", "nass", "bunt", "modern", "altmodisch", "teuer", "billig", "kostbar", "wertlos", "neu", "gebraucht", "kaputt", "ganz", "fest", "locker", "stabil", "wackelig", "rund", "eckig", "glatt", "rau", "scharf", "stumpf", "süß", "sauer", "salzig", "bitter", "würzig", "lecker", "eklig", "interessant", "langweilig", "spannend", "öde", "wichtig", "unwichtig", "nützlich", "nutzlos", "leicht", "schwer", "einfach", "kompliziert", "möglich", "unmöglich", "richtig", "falsch", "bekannt", "unbekannt", "berühmt", "fremd", "vertraut", "nah", "fern", "hoch", "tief", "breit", "schmal", "eng", "weit", "leer", "voll"]
ADJ_O = ["groß", "klein", "neu", "alt", "teuer", "billig", "hell", "dunkel", "warm", "kalt", "sauber", "schmutzig", "schön", "hässlich", "kaputt", "nützlich", "modern", "schwer", "leicht", "hart", "weich", "rund", "eckig", "bunt", "glatt", "rau", "fest", "locker", "stabil", "wackelig"]

PLACES_GEN = ["im Zimmer", "auf dem Tisch", "unter dem Bett", "im Büro", "in der Küche", "im Schrank", "im Regal", "in der Schachtel", "im Haus", "im Auto", "im Bad", "im Keller", "im Bad", "im Hotel", "im Kino", "im Museum", "im Theater", "im Laden", "im Restaurant", "im Garten", "in der Schule", "im Park", "in der Stadt", "im Wald", "am Bahnhof", "am Strand", "auf der Straße", "im Dorf", "auf dem Feld", "am See", "auf dem Berg", "auf der Wiese", "an der Brücke", "auf der Insel", "im Hof", "am Hafen", "auf dem Platz", "am Weg", "im Ozean", "am Flughafen", "auf der Terrasse", "am Flussufer", "auf dem Gipfel", "im Tal", "am Hang", "im Waldstadion", "auf dem Markt", "an der Haltestelle", "an der Station"]
EVENTS = ["die Schule", "der Film", "das Spiel", "das Essen", "die Party", "der Kurs", "das Treffen", "die Reise", "das Konzert", "die Arbeit", "der Urlaub", "das Frühstück", "die Prüfung", "der Ausflug", "das Abendessen", "der Termin", "die Hochzeit", "der Sport", "die Konferenz", "der Markt", "der Unterricht", "die Vorlesung", "die Messe", "der Geburtstag", "das Fest", "die Feier"]
TIMES = ["heute", "morgen", "am Montag", "am Dienstag", "am Mittwoch", "am Donnerstag", "am Freitag", "am Samstag", "am Sonntag", "am Wochenende", "um acht Uhr", "um zehn Uhr", "im Sommer", "im Winter", "bald", "jetzt", "später", "um zwei Uhr", "um vier Uhr", "nächste Woche", "nächsten Monat", "heute Abend", "morgen früh", "gleich", "sofort", "heute Nachmittag", "gestern", "vorgestern", "übermorgen", "in einer Stunde", "in zwei Tagen"]

VERBS_TRANS = ["liest", "kocht", "schreibt", "malt", "isst", "trinkt", "hört", "sieht", "sucht", "findet", "putzt", "wäscht", "zeichnet", "rechnet", "schneidet", "klebt", "bastelt", "backt", "brät", "kauft", "verkauft", "bringt", "holt", "trägt", "baut", "füttert", "streichelt", "besucht", "repariert", "öffnet", "schließt", "liefert"]
VERBS_INTRANS = ["arbeitet", "lernt", "singt", "tanzt", "geht", "läuft", "spielt", "lacht", "wartet", "schläft", "denkt", "telefoniert", "studiert", "übt", "probt", "zählt", "glaubt", "hofft", "liebt", "hasst", "versteht", "erklärt", "fragt", "erzählt", "berichtet", "riecht", "schmeckt", "fühlt", "wohnt"]
VERBS_MOVE_PEOPLE = ["kommt", "geht", "fährt", "reist", "fliegt", "läuft", "springt", "schwimmt", "wandert", "spaziert", "rennt", "klettert", "segelt", "taucht", "reitet"]
VERBS_MOVE_OBJECTS = ["gleitet", "rollt", "fließt", "weht"]
VERBS_ACT = VERBS_TRANS + VERBS_INTRANS + VERBS_MOVE_PEOPLE

VERB_OBJECT_MAP = {
    "liest": ["ein Medium", "ein Schreibmedium", "ein Wort", "ein Zeichen"],
    "kocht": ["eine Speise", "ein Gericht", "ein Lebensmittel", "ein Kochgeschirr", "ein Gemüse", "ein Fleischgericht"],
    "backt": ["ein Gebäck", "ein Lebensmittel", "ein Brot", "ein Kuchen"],
    "brät": ["ein Lebensmittel", "ein Nutztier", "ein Fleischgericht", "ein Fisch", "ein Gemüse"],
    "isst": ["ein Lebensmittel", "eine Frucht", "ein Gebäck", "ein Gericht", "ein Nutztier", "ein Gemüse", "ein Brot", "ein Kuchen", "ein Fisch"],
    "trinkt": ["ein Getränk", "ein Gewässer", "Saft", "Kaffee", "Wein", "Tee", "Milch", "Bier"],
    "schreibt": ["ein Medium", "ein Schreibmedium", "ein Wort", "ein Zeichen", "ein Brief", "eine Zeitung"],
    "malt": ["ein Kunstwerk", "ein Bild", "ein Gegenstand", "ein Tier", "eine Person", "eine Pflanze", "ein Gebäude"],
    "putzt": ["ein Möbelstück", "ein Gerät", "ein Fahrzeug", "ein Geschirr", "ein Besteck", "ein Bauelement", "ein Raum", "ein Reinigungsgerät", "ein Gegenstand"],
    "wäscht": ["ein Kleidungsstück", "ein Textil", "ein Geschirr", "ein Besteck", "ein Tier", "eine Person", "ein Fahrzeug", "ein Gegenstand"],
    "zeichnet": ["ein Kunstwerk", "ein Bild", "ein Gegenstand", "ein Tier", "eine Pflanze"],
    "schneidet": ["ein Lebensmittel", "ein Textil", "ein Material", "eine Pflanze", "ein Baustoff", "ein Papier", "ein Holz"],
    "bastelt": ["ein Gegenstand", "ein Spielzeug", "ein Kunstwerk", "ein Modell"],
    "trägt": ["ein Kleidungsstück", "ein Accessoire", "eine Kopfbedeckung", "eine Tasche", "eine Sehhilfe", "ein Schmuckstück"],
    "baut": ["ein Gebäude", "ein Bauwerk", "ein Möbelstück", "ein Fahrzeug", "ein Gerät", "ein Zelt", "ein Boot"],
    "besucht": ["ein Ort", "eine Person", "ein Gebäude", "eine Einrichtung", "ein Berg", "ein Museum", "ein Theater", "ein Park", "eine Stadt"],
    "repariert": ["ein Gerät", "ein Fahrzeug", "ein Möbelstück", "ein Bauelement", "ein Werkzeug", "ein Computer", "ein Wagen", "ein Fahrrad"],
    "öffnet": ["ein Bauelement", "ein Behälter", "ein Gebäude", "ein Raum", "ein Buch", "eine Tür", "ein Fenster", "ein Schrank", "ein Koffer"],
    "schließt": ["ein Bauelement", "ein Behälter", "ein Gebäude", "ein Raum", "ein Buch", "eine Tür", "ein Fenster", "ein Schrank", "ein Koffer"],
    "füttert": ["ein Tier", "ein Haustier", "ein Wildtier", "ein Nutztier", "ein Vogel", "ein Fisch", "ein Hund", "eine Katze", "ein Pferd"],
    "streichelt": ["ein Tier", "ein Haustier", "ein Wildtier", "ein Nutztier", "eine Person", "eine Katze", "ein Hund", "ein Pferd", "ein Schaf"],
    "hört": ["ein Medium", "ein Musikinstrument", "ein Geräusch", "ein Tier", "eine Person", "Musik", "ein Lied", "ein Radio"],
    "studiert": ["ein Begriff", "ein Medium", "ein Buch", "eine Sprache", "ein Fach"],
    "rechnet": ["ein Begriff", "eine Zahl", "eine Aufgabe"],
    "singt": ["ein Lied", "ein Medium", "Musik"],
    "lernt": ["ein Begriff", "ein Wort", "ein Medium", "ein Buch", "eine Sprache"],
    "hat": ["ein Gegenstand", "ein Gerät", "ein Tier", "ein Lebensmittel", "ein Besteck", "ein Möbelstück", "ein Kleidungsstück", "ein Fahrzeug", "ein Spielzeug", "ein Accessoire", "eine Kopfbedeckung", "eine Tasche", "eine Sehhilfe", "ein Objekt", "ein Ding", "ein Fragment", "eine Pflanze", "eine Frucht", "ein Gewässer"],
    "sucht": ["ein Gegenstand", "ein Gerät", "ein Tier", "eine Person", "ein Ort", "ein Bauelement", "ein Möbelstück", "ein Fahrzeug", "ein Objekt", "ein Ding", "ein Fragment", "ein Schlüssel", "ein Buch"],
    "findet": ["ein Gegenstand", "ein Gerät", "ein Tier", "eine Person", "ein Ort", "ein Bauelement", "ein Möbelstück", "ein Fahrzeug", "ein Objekt", "ein Ding", "ein Fragment", "ein Schlüssel", "ein Buch"],
    "sieht": ["ein Gegenstand", "ein Gerät", "ein Tier", "eine Person", "ein Ort", "ein Bauelement", "ein Möbelstück", "ein Fahrzeug", "ein Objekt", "ein Ding", "ein Fragment", "ein Berg", "ein See", "ein Haus"],
}

def is_semantic_match(v, n):
    if v not in VERB_OBJECT_MAP: return True
    allowed = VERB_OBJECT_MAP[v]
    cat = CAT_MAP.get(n, "")
    return any(a in cat or cat in a for a in allowed)

def is_subject_match(v, n):
    living = NOUNS_DB[n]['l']
    if v in VERBS_MOVE_OBJECTS:
        return not living
    if v in VERBS_ACT or v in VERBS_MOVE_PEOPLE:
        return living
    return True

ROLES = ["mein Freund", "meine Schwester", "der Lehrer", "die Ärztin", "mein Bruder", "eine Kollegin", "der Nachbar", "ein Kind", "der Koch", "die Pilotin", "mein Onkel", "meine Tante", "mein Opa", "meine Oma", "der Chef", "die Kollegin", "ein Bekannter", "die Cousine", "der Verwandte", "die Nachbarin", "der Schüler", "die Schülerin", "der Student", "die Studentin", "der Professor", "die Professorin", "der Polizist", "die Polizistin", "der Arzt", "die Krankenschwester", "der Ingenieur", "die Architektin", "der Autor", "der Maler", "die Sängerin", "der Sportler"]

def get_art(n, case="nom", cap=False, indefinite=False):
    info = NOUNS_DB[n]
    g = info['g']
    if indefinite:
        if case == "acc" and g == 0: art = "einen"
        elif case == "dat":
            art = ["einem", "einer", "einem"][g]
        else:
            art = ["ein", "eine", "ein"][g]
    else:
        if case == "acc":
            art = ["den", "die", "das"][g]
        elif case == "dat":
            art = ["dem", "der", "dem"][g]
        else:
            art = ["der", "die", "das"][g]
    return art.capitalize() if cap else art

def get_adj_decl(adj, color, n, case="nom"):
    info = NOUNS_DB[n]
    g = info['g']
    if color:
        if color in INDECL_COLORS: return color
        if color in ADJ_COLORS:
            base = ADJ_COLORS[color]
            if case == "acc" and g == 0: return base + "n"
            if case == "dat": return base + "n"
            return base
        suffix = "e"
        if case == "acc" and g == 0: suffix = "en"
        if case == "dat": suffix = "en"
        return color + suffix
    if adj:
        suffix = "e"
        if case == "acc" and g == 0: suffix = "en"
        if case == "dat": suffix = "en"
        return adj + suffix
    return ""

def safe_sample(l, k):
    return random.sample(l, min(len(l), k))

def generate_dataset(output_file, target_size=100000):
    dataset = []
    seen = set()
    counts = Counter()
    
    targets = {"WAS": 15000, "WER": 15000, "WO": 15000, "WANN": 15000, "WIE": 15000, "WARUM": 10000, "JA_NEIN": 15000}
    all_nouns = list(NOUNS_DB.keys())

    def add_p(q, a, t):
        if counts[t] >= targets[t]: return False
        if q not in seen and 10 <= len(a) <= 120:
            dataset.append({"question": q, "answer": a})
            seen.add(q)
            counts[t] += 1
            return True
        return False

    print("Step 1: Systematic Combinatorial Generation...")

    # WER
    print("Generating WER...")
    for name in safe_sample(NAMES, len(NAMES)):
        for role in safe_sample(ROLES, len(ROLES)):
            add_p(f"Wer ist {name}?", f"{name} ist {role}.", "WER")
    
    for v in safe_sample(VERBS_ACT, len(VERBS_ACT)):
        for loc in safe_sample(PLACES_GEN, len(PLACES_GEN)):
            add_p(f"Wer {v} {loc}?", f"{random.choice(NAMES)} {v} dort.", "WER")
    
    for n in safe_sample(all_nouns, min(2000, len(all_nouns))):
        art_acc = get_art(n, case="acc")
        for v in safe_sample(VERBS_TRANS, min(len(VERBS_TRANS), 50)):
            if is_semantic_match(v, n):
                add_p(f"Wer {v} {art_acc} {n}?", f"{random.choice(NAMES)} {v} {art_acc} {n}.", "WER")
    
    for loc in safe_sample(PLACES_GEN, len(PLACES_GEN)):
        add_p(f"Wer wohnt {loc}?", f"{random.choice(NAMES)} wohnt {loc}.", "WER")
    
    for role in safe_sample(ROLES, len(ROLES)):
        role_word = role.split()[-1]
        add_p(f"Wer arbeitet als {role_word}?", f"{random.choice(NAMES)} arbeitet als {role_word}.", "WER")

    # WIE
    print("Generating WIE...")
    for name in safe_sample(NAMES, min(2000, len(NAMES))):
        for adj in safe_sample(ADJ_P, min(len(ADJ_P), 100)):
            add_p(f"Wie ist {name}?", f"{name} ist {adj}.", "WIE")
            add_p(f"Wie geht es {name}?", f"Es geht {name} heute sehr {adj}.", "WIE")
    
    for name in safe_sample(NAMES, min(2000, len(NAMES))):
        add_p(f"Wie alt ist {name}?", f"{name} ist {random.randint(5,95)} Jahre alt.", "WIE")
    
    for n in safe_sample(all_nouns, min(2000, len(all_nouns))):
        art = get_art(n)
        for adj in safe_sample(ADJ_O, min(len(ADJ_O), 100)):
            add_p(f"Wie ist {art} {n}?", f"{art.capitalize()} {n} ist {adj}.", "WIE")

    # WAS
    print("Generating WAS...")
    for n in CAT_MAP:
        add_p(f"Was ist {get_art(n, indefinite=True)} {n}?", f"{get_art(n, cap=True, indefinite=True)} {n} ist {CAT_MAP[n]}.", "WAS")
    
    for n in safe_sample(all_nouns, min(2000, len(all_nouns))):
        for adj in safe_sample(ADJ_O, min(len(ADJ_O), 100)):
            decl_adj = get_adj_decl(adj, None, n)
            add_p(f"Was ist {decl_adj}?", f"{get_art(n, cap=True)} {n} ist {adj}.", "WAS")  # Predicative: undeclined
        for col in safe_sample(list(ADJ_COLORS.keys()) + INDECL_COLORS, min(100, len(ADJ_COLORS)+len(INDECL_COLORS))):
            decl_col = get_adj_decl(None, col, n)
            add_p(f"Welche Farbe hat {get_art(n)} {n}?", f"{get_art(n, cap=True)} {n} ist {col}.", "WAS")  # Predicative: undeclined

    # WO
    print("Generating WO...")
    for n in safe_sample(all_nouns, min(2000, len(all_nouns))):
        for loc in safe_sample(PLACES_GEN, min(len(PLACES_GEN), 100)):
            add_p(f"Wo ist {get_art(n)} {n}?", f"{get_art(n, cap=True)} {n} ist {loc}.", "WO")
    
    for v in safe_sample(VERBS_ACT, min(len(VERBS_ACT), 100)):
        for name in safe_sample(NAMES, min(200, len(NAMES))):
            add_p(f"Wo {v} {name}?", f"{name} {v} {random.choice(PLACES_GEN)}.", "WO")

    # WANN
    print("Generating WANN...")
    for ev in safe_sample(EVENTS, len(EVENTS)):
        for t in safe_sample(TIMES, min(len(TIMES), 100)):
            add_p(f"Wann beginnt {ev}?", f"{ev.capitalize()} beginnt {t}.", "WANN")
    
    for name in safe_sample(NAMES, min(1000, len(NAMES))):
        for v in safe_sample(VERBS_MOVE_PEOPLE, len(VERBS_MOVE_PEOPLE)):
            add_p(f"Wann {v} {name}?", f"{name} {v} {random.choice(TIMES)}.", "WANN")

    # WARUM
    print("Generating WARUM...")
    for name in safe_sample(NAMES, min(1000, len(NAMES))):
        for adj in safe_sample(ADJ_P, min(len(ADJ_P), 100)):
            add_p(f"Warum ist {name} {adj}?", f"Weil {name} heute viel Stress hatte.", "WARUM")
    
    for n in safe_sample(all_nouns, min(1000, len(all_nouns))):
        for adj in safe_sample(ADJ_O, min(len(ADJ_O), 100)):
            add_p(f"Warum ist {get_art(n)} {n} {adj}?", f"Weil {get_art(n)} {n} schon sehr alt ist.", "WARUM")

    # JA_NEIN
    print("Generating JA_NEIN...")
    for n in safe_sample(all_nouns, min(2000, len(all_nouns))):
        for col in safe_sample(list(ADJ_COLORS.keys()) + INDECL_COLORS, min(100, len(ADJ_COLORS)+len(INDECL_COLORS))):
            d_col = get_adj_decl(None, col, n)
            loc = random.choice(PLACES_GEN)
            add_p(f"Ist {get_art(n)} {d_col} {n} {loc}?", f"Ja, {get_art(n)} {d_col} {n} ist {loc}.", "JA_NEIN")
    
    for name in safe_sample(NAMES, min(1000, len(NAMES))):
        for n in safe_sample(all_nouns, min(500, len(all_nouns))):
            if is_semantic_match("hat", n):
                add_p(f"Hat {name} {get_art(n, case='acc', indefinite=True)} {n}?", f"Ja, {name} hat {get_art(n, case='acc', indefinite=True)} {n}.", "JA_NEIN")

    print("Step 2: Strict Force Gap Filling...")
    iters = 0
    while any(counts[t] < targets[t] for t in targets) and iters < 50000000:
        iters += 1
        needed = [t for t in targets if counts[t] < targets[t]]
        if not needed: break
        
        qt = random.choice(needed)
        n = random.choice(all_nouns)
        name_v = random.choice(NAMES)
        col = random.choice(list(ADJ_COLORS.keys()) + INDECL_COLORS)
        adj_o = random.choice(ADJ_O)
        adj_p = random.choice(ADJ_P)
        loc = random.choice(PLACES_GEN)
        t_v = random.choice(TIMES)
        v_act = random.choice(VERBS_ACT)
        ev = random.choice(EVENTS)
        v_trans = random.choice(VERBS_TRANS)
        v_intrans = random.choice(VERBS_INTRANS)
        v_move_p = random.choice(VERBS_MOVE_PEOPLE)
        v_move_o = random.choice(VERBS_MOVE_OBJECTS)
        role = random.choice(ROLES)
        
        r = random.random()
        name_v2 = random.choice(NAMES)  # Second name for variety
        if qt == "WER":
            # More diverse WER templates with random elements
            templates = [
                (f"Wer ist {name_v}?", f"{name_v} ist {role}."),
                (f"Wer {v_act} {loc}?", f"{name_v} {v_act} dort."),
                (f"Wer wohnt {loc}?", f"{name_v} wohnt dort."),
                (f"Wer arbeitet als {role.split()[-1]}?", f"{name_v} arbeitet als {role.split()[-1]}."),
                (f"Wer kommt {t_v}?", f"{name_v} kommt {t_v}."),
                (f"Wer singt {t_v}?", f"{name_v} singt {t_v}."),
                (f"Wer tanzt {t_v}?", f"{name_v} tanzt {t_v}."),
                (f"Wer lacht {loc}?", f"{name_v} lacht dort."),
                (f"Wer wartet {loc}?", f"{name_v} wartet dort."),
                (f"Wer schläft {t_v}?", f"{name_v} schläft {t_v}."),
                (f"Wer {v_intrans} {t_v}?", f"{name_v} {v_intrans} {t_v}."),
                (f"Wer {v_intrans} {loc}?", f"{name_v} {v_intrans} dort."),
                (f"Wer kennt {name_v2}?", f"{name_v} kennt {name_v2}."),
                (f"Wer besucht {name_v2} {t_v}?", f"{name_v} besucht {name_v2} {t_v}."),
                (f"Wer hilft {name_v2}?", f"{name_v} hilft {name_v2}."),
                (f"Wer ruft {name_v2} an?", f"{name_v} ruft {name_v2} an."),
                (f"Wer trifft {name_v2} {t_v}?", f"{name_v} trifft {name_v2} {t_v}."),
                (f"Wer spricht mit {name_v2}?", f"{name_v} spricht mit {name_v2}."),
                (f"Wer fährt {t_v}?", f"{name_v} fährt {t_v}."),
                (f"Wer reist {t_v}?", f"{name_v} reist {t_v}."),
            ]
            q, a = random.choice(templates)
            add_p(q, a, "WER")
        elif qt == "WIE":
            if r < 0.2: add_p(f"Wie geht es {name_v}?", f"Es geht {name_v} heute sehr {adj_p}.", "WIE")
            elif r < 0.4: add_p(f"Wie ist {get_art(n)} {n}?", f"{get_art(n, cap=True)} {n} ist {adj_o}.", "WIE")
            elif r < 0.6: add_p(f"Wie {v_intrans} {name_v}?", f"{name_v} {v_intrans} sehr {adj_p}.", "WIE")
            elif r < 0.8: add_p(f"Wie alt ist {name_v}?", f"{name_v} ist {random.randint(1,99)} Jahre alt.", "WIE")
            else: add_p(f"Wie schnell {v_intrans} {name_v}?", f"{name_v} {v_intrans} ziemlich {adj_p}.", "WIE")
        elif qt == "WAS":
            if r < 0.1: add_p(f"Was ist {adj_o}?", f"{get_art(n, cap=True)} {n} ist {adj_o}.", "WAS")
            elif r < 0.2: 
                if is_semantic_match(v_trans, n):
                    add_p(f"Was {v_trans} {name_v}?", f"{name_v} {v_trans} {get_art(n, case='acc')} {n}.", "WAS")
            elif r < 0.3: add_p(f"Was ist {get_art(n, indefinite=True)} {n}?", f"{get_art(n, cap=True, indefinite=True)} {n} ist {CAT_MAP[n]}.", "WAS")
            elif r < 0.4: 
                if is_semantic_match("sucht", n):
                    add_p(f"Was sucht {name_v}?", f"{name_v} sucht {get_art(n, case='acc')} {n}.", "WAS")
            elif r < 0.5:
                decl_col = get_adj_decl(None, col, n)
                add_p(f"Was ist {decl_col}?", f"{get_art(n, cap=True)} {n} ist {col}.", "WAS")  # Predicative: undeclined
            elif r < 0.6:
                if is_semantic_match("findet", n):
                    add_p(f"Was findet {name_v} {loc}?", f"{name_v} findet dort {get_art(n, case='acc', indefinite=True)} {n}.", "WAS")
            elif r < 0.7:
                if is_semantic_match(v_trans, n):
                    add_p(f"Was {v_trans} {name_v} {t_v}?", f"{name_v} {v_trans} {t_v} {get_art(n, case='acc')} {n}.", "WAS")
            elif r < 0.8:
                add_p(f"Was kocht {name_v} {t_v}?", f"{name_v} kocht {t_v} {random.choice(['eine Suppe', 'ein Gericht'])}.", "WAS")
            elif r < 0.9:
                if is_semantic_match("baut", n):
                    add_p(f"Was baut {name_v} {loc}?", f"{name_v} baut dort {get_art(n, case='acc', indefinite=True)} {n}.", "WAS")
            else:
                if is_semantic_match("liest", n):
                    add_p(f"Was liest {name_v} {t_v}?", f"{name_v} liest {t_v} {get_art(n, case='acc', indefinite=True)} {n}.", "WAS")
        elif qt == "WO":
            if r < 0.33: add_p(f"Wo ist {get_art(n)} {n}?", f"{get_art(n, cap=True)} {n} ist {loc}.", "WO")
            elif r < 0.66: add_p(f"Wo {v_act} {name_v}?", f"{name_v} {v_act} {loc}.", "WO")
            else: add_p(f"Wo wohnt {name_v}?", f"{name_v} wohnt {loc}.", "WO")
        elif qt == "WANN":
            if r < 0.33: add_p(f"Wann beginnt {ev}?", f"{ev.capitalize()} beginnt {t_v}.", "WANN")
            elif r < 0.66: add_p(f"Wann {v_move_p} {name_v}?", f"{name_v} {v_move_p} {t_v}.", "WANN")
            else: add_p(f"Wann {v_act} {name_v}?", f"{name_v} {v_act} {t_v}.", "WANN")
        elif qt == "WARUM":
            if r < 0.5: add_p(f"Warum ist {name_v} {adj_p}?", f"Weil {name_v} heute viel Stress hatte.", "WARUM")
            else: add_p(f"Warum {v_intrans} {name_v}?", f"Weil {name_v} es so gelernt hat.", "WARUM")
        elif qt == "JA_NEIN":
            if r < 0.5: 
                if is_semantic_match("hat", n):
                    add_p(f"Hat {name_v} {get_art(n, case='acc', indefinite=True)} {n}?", f"Ja, {name_v} hat {get_art(n, case='acc', indefinite=True)} {n}.", "JA_NEIN")
            else: add_p(f"Ist {get_art(n)} {n} {loc}?", f"Ja, {get_art(n)} {n} ist {loc}.", "JA_NEIN")

    print(f"Final Counts: {counts}")
    print(f"Total Unique Pairs Generated: {len(dataset)}")
    
    random.shuffle(dataset)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset[:target_size]:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"Success: {output_file} created.")

if __name__ == "__main__":
    generate_dataset("german_qa_dataset.jsonl", 100000)
