import re
import string

import spacy

from .constants import DAYS, MONTHS, NAMES, USCITIES, NA_STATES, COUNTRIES, NATIONALITIES

PUNCTUATION_SPACE_PATTERN = re.compile(r'(\s+[{}])'.format(
    ''.join(re.escape(p) for p in string.punctuation if p != '\\')))

WORDS_TO_CAPITALIZE_MAP = {word.lower(): word for word in set([w.capitalize() for w in {'i'} | DAYS | MONTHS])
                           | NAMES | USCITIES | NA_STATES | COUNTRIES | NATIONALITIES}

WORDS_TO_CAPITALIZE_STOP_WORDS = {'mobile'}
for word in WORDS_TO_CAPITALIZE_STOP_WORDS:
    del WORDS_TO_CAPITALIZE_MAP[word]

WORDS_TO_CAPITALIZE_PATTERN = re.compile(r'\b({})\b'.format('|'.join(WORDS_TO_CAPITALIZE_MAP.keys())))

CONTRACTION_SUFFIX_PATTERN = re.compile(r'\b\s+({})\b'.format('|'.join(
    (
        "n\'t",
        "\'re"
        "\'s",
        "\'d",
        "\'ll"
        "\'t",
        "\'ve",
        "\'m",
    )
)))

TERMINAL_PUNCTUATION = list("""!#$%&'*+,-./:;<=>?@~""")

SPACY_PARSER = spacy.load('en_core_web_sm', diable=['ner', 'tagger'])


def postprocess(decoded_tokens):
    for dec in decoded_tokens:
        while dec[0] in TERMINAL_PUNCTUATION:
            dec.pop(0)

    decoded_sentences = [' '.join(dec) for dec in decoded_tokens]
    capitalized_decoded_sentences = []
    for doc in SPACY_PARSER.pipe(decoded_sentences):
        capitalized_decoded_sentences.append(' '.join(sent.text.capitalize() for sent in doc.sents))
    decoded_sentences = capitalized_decoded_sentences

    for i in range(len(decoded_sentences)):
        decoded_sentences[i] = PUNCTUATION_SPACE_PATTERN.sub(lambda x: x.group(1).replace(" ", ""),
                                                             decoded_sentences[i])
        decoded_sentences[i] = WORDS_TO_CAPITALIZE_PATTERN.sub(lambda x: WORDS_TO_CAPITALIZE_MAP[x.group(1)],
                                                               decoded_sentences[i])
        decoded_sentences[i] = CONTRACTION_SUFFIX_PATTERN.sub(lambda x: x.group(1).replace(" ", ""),
                                                              decoded_sentences[i])

    return decoded_sentences
