import nltk
from nltk.corpus import names, gazetteers

for data in ('names', 'gazetteers'):
    nltk.download(data, quiet=True)

DAYS = {'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}

MONTHS = {'january', 'february', 'march', 'april', 'may', 'june', 'july',
          'august', 'september', 'october', 'november', 'december',
          'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept',
          'oct', 'nov', 'dec'}

NAMES = set([name for filename in ('male.txt', 'female.txt') for name
             in names.words(filename)])

USCITIES = set([city for city in gazetteers.words('uscities.txt')])

# [XX] contains some non-ascii chars
COUNTRIES = set([country for filename in ('isocountries.txt', 'countries.txt')
                 for country in gazetteers.words(filename)])

# States in North America
NA_STATES = set([state for filename in ('usstates.txt', 'mexstates.txt', 'caprovinces.txt')
                 for state in gazetteers.words(filename)])

NATIONALITIES = set([nationality for nationality in gazetteers.words('nationalities.txt')])
