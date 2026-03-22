import random

def synonym_replace(text):
    synonyms = {
        "fever": "high temperature",
        "cough": "throat irritation",
        "pain": "discomfort",
        "headache": "head pain",
        "fatigue": "tiredness",
        "nausea": "feeling sick",
        "vomiting": "throwing up",
        "rash": "skin irritation",
        "swelling": "inflammation",
        "dizziness": "lightheadedness",
        "shortness": "difficulty",
        "breathing": "respiration",
        "chest": "thoracic",
        "stomach": "abdominal",
        "throat": "pharynx",
        "weakness": "lack of strength",
        "infection": "illness",
        "anxiety": "nervousness",
        "depression": "low mood",
        "itching": "pruritus",
    }
    words = text.split()
    return " ".join([synonyms.get(w, w) for w in words])

def random_typo(text, prob=0.1):
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < prob:
            chars[i] = random.choice("abcdefghijklmnopqrstuvwxyz")
    return "".join(chars)


def drop_words(text, prob=0.1):
    words = text.split()
    return " ".join([w for w in words if random.random() > prob])
