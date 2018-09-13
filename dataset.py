import unicodedata
import torch

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Lang(object):
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Convert Unicode to Ascii
def unicodeToAscii(s):
    chars = [c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn']
    char_list = ''.join(chars)
    return char_list

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    return s

def readLangs(lang1, lang2, reverse=False):
    #print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')

    #print("Printing Lines...")
    #print(lines)

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    #print("Printing Pairs...")
    #print(pairs)

    #Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    #print("Printing Pairs...")
    #print(pairs)
    return input_lang, output_lang, pairs


def filterPair(p):
    is_good_length = len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH
    return is_good_length

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)

    #print("Read %s sentence pairs" % len(pairs))

    pairs = filterPairs(pairs)

    #print("Pair's After Trimming")
    #print(pairs)

    #print("Trimmed to %s sentence pairs" % len(pairs))
    #print("Counting words...")

    for pair in pairs:
        #print(pair[0])
        #print(pair[1])
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    #print("Counted words:")
    #print(input_lang.name, input_lang.n_words)
    #print(output_lang.name, output_lang.n_words)
    #print(random.choice(pairs))
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long,device=device).view(-1, 1)


def tensorsFromPair(input_lang, output_lang,pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)