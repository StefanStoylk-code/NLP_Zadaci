"""
Kreirati „naivni“ sistem za odgovaranje na pitanja. 
Sistem treba da odgovara na pitanje tipa „Ko...“, „Gde...“ i „Kada...“. 
Korišćenjem NLTK alata izvršiti tokenizaciju, segmenataciju rečenica i izvršiti prepoznavanje osoba, lokacija i datuma. 
Izdvojiti rečenice koje sadrže reči iz pitanja i reči sa odgovarajućim tagom (osoba, lokacija ili datum). 
Kreirati odgovore korišćenjem izdvojenih podataka.
"""

# Uvoz potrebnih alata
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

# Tekst na kom ce biti predstavljen naivni sistem za odgovaranje na pitanja
text = """ Nikola Tesla was born on 10 July 1856 in Smiljan, a village in the Austrian Empire. He studied engineering at the Polytechnic in Graz. In 1884, Tesla emigrated to the United States and arrived in New York.
He briefly worked for Thomas Edison at the Edison Machine Works. George Westinghouse bought Tesla's patents for alternating current motors in 1888. In 1893, Westinghouse and Tesla lit the World's Columbian Exposition in Chicago using alternating current.
Tesla began building the Wardenclyffe Tower on Long Island in 1901. Tesla died on 7 January 1943 in New York City. The Nikola Tesla Museum was founded in Belgrade in 1952.
"""

print("==== Tekst primer ====")
TEXT = text.strip()
print(TEXT)
print()

print("==== Tokenizacija teksta ====")
TOKENES = word_tokenize(TEXT)
print(TOKENES)
print()

print("==== POS tagovanje teksta ====")
POS_TAG = pos_tag(TOKENES)
print(POS_TAG)
print()

# Prepoznavanje datuma
DAY = r"\d{1,2}(?:st|nd|rd|th)?"
MONTHS = (r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?") 
YEAR = r"(?:1\d{3}|20\d{2})" 

DATE_PATTERN = re.compile(
    rf"\b(?:"
    rf"{DAY}\s+(?:of\s+)?{MONTHS},?\s+{YEAR}"    # 10 July 1856
    rf"|{MONTHS}\s+{DAY},?\s+{YEAR}"             # July 10, 1856
    rf"|{MONTHS}\s+{YEAR}"                       # July 1856
    rf"|\d{{1,2}}[./]\d{{1,2}}[./]\d{{2,4}}"     # 10.07.1856 or 10/07/1856
    rf"|\d{{4}}-\d{{2}}-\d{{2}}"                 # 1856-07-10
    rf"|{DAY}\s+(?:of\s+)?{MONTHS}"              # 10 July
    rf"|{MONTHS}\s+{DAY}"                        # July 10
    rf"|{YEAR}s?"                                # 1856, 1890s
    rf")\b"
)

# Pronalazi datum entitete u tekstu
def recognize_dates(tagged_tokens):
    text = " ".join(word for word, tag in tagged_tokens)
    text = re.sub(r"\s+([,.])", r"\1", text)   

    return [("DATE", m.group(0)) for m in DATE_PATTERN.finditer(text)]

# Prepoznavanje imenovanih entiteta: NLTK: PERSON, GPE, LOCATION, ... + regex datum
def recognize_entities(tagged_tokens):
    entities = []  # Lista tuplova tipa (label, entity_text)
    tree = ne_chunk(tagged_tokens)  # NLTK-ov canker imenovanih entiteta

    for node in tree:
        if isinstance(node, Tree):
            label = node.label()
            entity_text = " ".join(word for word, tag in node.leaves())
            entities.append((label, entity_text))

    entities.extend(recognize_dates(tagged_tokens))

    return entities

print("==== Segmentacija teksta ====")
SEGMENTED_TEXT = sent_tokenize(TEXT)
for i, s in enumerate(SEGMENTED_TEXT, 1):
    print(f"{i}. {s}")  # Redni broj recenice + recenica

# Obrada teksta
def process_text(sentences):
    processed = []  # Lista u kojoj ce biti sacuvana recenica, njeni tokeni i entiteti

    for sentence in sentences:
        TOKENS = word_tokenize(sentence)          # Tokenizacija
        POS_TAGGED = pos_tag(TOKENS)              # POS tagovanje
        entities = recognize_entities(POS_TAGGED) # Prepoznavanje entiteta
        processed.append({
            "sentence": sentence,
            "tokens": TOKENS,
            "entities": entities,
        })

    return processed

print()
print("==== Obradjen tekst ====")
PROCESSED_TEXT = process_text(SEGMENTED_TEXT)
for p in PROCESSED_TEXT:
    print(p["sentence"])
    print("   entities:", p["entities"])
print()

# Odredjivanje pitanja
def determine_question_type(question):
    question_tokens = word_tokenize(question)  # Tokenizacija pitanja 

    if not question_tokens:  
        return None, question_tokens

    # Pitanje mora poceti sa 'who', 'when' ili 'where'
    first = question_tokens[0].lower()
    if first == "who":
        question_type = "PERSON"
    elif first == "where":
        question_type = "LOCATION"
    elif first == "when":
        question_type = "DATE"
    else:
        question_type = None  # Nepodrzano pitanje

    return question_type, question_tokens  # Vraca pitanje sa svojim tokenima

QUESTION_WORDS = {"who", "where", "when", "is", "was", "were", "are", "did", "do", "does", "whether", "which", "what", "whom", "whose"}  # Upitne reci
STOP_WORDS = set(stopwords.words("english")) | QUESTION_WORDS  # Stop reci sa upitnim recima

# Mala slova + stemovanje, uklanjanje posesivnog nastavka „'s”.
def normalize(token):
    token = token.lower()

    if token in ("'s", "’s"):
        return ""
    
    return stemmer.stem(token)

# Odabir kljucnih reci iz pitanja
def extract_keywords(question_tokens):
    keywords = []  # Lista kljucnih reci

    for token in question_tokens:
        t = token.lower()
        if t in STOP_WORDS or not any(ch.isalnum() for ch in t):
            continue  # Uklanja upitne reci, ceste reci i interpunkciju
        stem = normalize(t)
        if stem and stem not in keywords:
            keywords.append(stem)

    return keywords

# NLTK oznake entiteta koji odgovaraju pojedinim tipovima pitanja
ENTITY_TYPES = {
    "PERSON":   {"PERSON"},
    "LOCATION": {"GPE", "LOCATION", "GSP"},
    "DATE":     {"DATE"},
}

# Pronalazenje recenice za odgovor na pitanje
def find_candidates(processed_sentences, keywords, question_type):
    candidates = []  # Lista recenica kao kandidati za odgovor

    for p in processed_sentences:
        sentence_stems = {normalize(t) for t in p["tokens"]}

        number_of_matches = 0
        for keyword in keywords:
            if keyword in sentence_stems:
                number_of_matches += 1

        corresponding_entities = [e for e in p["entities"] if e[0] in ENTITY_TYPES[question_type]]

        if number_of_matches > 0 and corresponding_entities:
            candidates.append({**p, "matches": number_of_matches,
                               "corresponding_entities": corresponding_entities})
            
    return candidates

# Odabir najboljeg kandidata
def choose_best_candidate(candidates):
    if not candidates:
        return None
    
    return max(candidates, key=lambda c: c["matches"])

#  Odabir odgovora
def select_answer(candidate, question_type, keywords):
    corresponding_entities = [text for label, text in candidate["entities"] if label in ENTITY_TYPES[question_type]]

    if not corresponding_entities:
        return None

    # Preferirati entitet koji ne ponavlja reci iz pitanja
    for entity in corresponding_entities:
        entity_stems = {normalize(w) for w in entity.split()}
        if not entity_stems & set(keywords):
            return entity

    # U suprotnom: prvi entitet odgovarajucceg tipa
    return corresponding_entities[0]


# Pipeline funkcija
def answer_question(question, processed_sentences, verbose=True):
    question_type, question_tokens = determine_question_type(question)  # Tokenizacija pitanja

    if question_type is None:
        result = {"question": question, 
                  "sentence": None, 
                  "answer": "Unsupported question type"}
    else:
        keywords = extract_keywords(question_tokens)  # Trazi kljucne reci
        candidates = find_candidates(processed_sentences, keywords, question_type)  # Trazi kandidata
        best = choose_best_candidate(candidates)  # Uzima najboljeg kandidata
        if best is None:
            result = {"question": question, 
                      "sentence": None, 
                      "answer": "Answer not found"}
        else:
            result = {"question": question, 
                      "sentence": best["sentence"],
                      "answer": select_answer(best, question_type, keywords)}
            
    if verbose:  # Stampanje rezultata
        print("==== Rezultat ====")
        print("Question:          ", result["question"])

        if result["sentence"]:
            print("Relevant sentence: ", result["sentence"])

        print("Answer:            ", result["answer"])
        print("-" * 70)

    return result

# Lista pitanja 
QUESTIONS = [
    "Where was Nikola Tesla born?",
    "When was Nikola Tesla born?",
    "Who bought Tesla's patents?",
    "When did Tesla emigrate to the United States?",
    "Where did Tesla study engineering?",
    "Where did Tesla die?",
    "When did Tesla die?",
    "Where was the Nikola Tesla Museum founded?",
    "Who worked with Tesla to light the exposition in Chicago?",
    "What is Tesla's most prominent patent?",      # Pitanje za koje nema odgovora u tekst primeru
]

# Pokretanje pipeline funkcije
results = [answer_question(q, PROCESSED_TEXT) for q in QUESTIONS]
