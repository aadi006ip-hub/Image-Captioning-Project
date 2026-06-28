from collections import Counter
from src.data_preprocessing import TextPreprocessor

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self): return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in TextPreprocessor.clean_caption(sentence).split(' '):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word], self.itos[idx] = idx, word
                    idx += 1

    def numericalize(self, text):
        tokenized = TextPreprocessor.clean_caption(text).split(' ')
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized]