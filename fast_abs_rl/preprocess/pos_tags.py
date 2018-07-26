from scipy.stats import chisquare

POS_TAGS = [
    'ADJ',
    'ADP',
    'ADV',
    'AUX',
    'CONJ',
    'CCONJ',
    'DET',
    'INTJ',
    'NOUN',
    'NUM',
    'PART',
    'PRON',
    'PROPN',
    'PUNCT',
    'SCONJ',
    'SYM',
    'VERB',
    'X',
    'SPACE',
]


def smooth(nums, epsilon=1e-6):
    return [max(num, epsilon) for num in nums]


def compute_pos_tag_chisquare(tag_obs, tag_exp):
    f_obs = [tag_obs[tag] for tag in POS_TAGS]
    f_exp = smooth([tag_exp[tag] for tag in POS_TAGS])
    return chisquare(f_obs, f_exp)
