"""Byte Pair Encoding Tokenizer

Def: vocab (lowest level unit)
Def: word (list of one or more items from vocab)

Psuedo code:
words = get_words(data)
vocab_freq = get_init_vocab()
vocab_freq


"""

from typing import List


class Tokenizer:
    def __init__(self, data: List[List[str]]) -> None:
        # create word set
        self.words = set()
        for s in data:
            for w in s:
                self.words.add(w)

        # create vocab
        self.vocab = self._get_init_vocab(self.words)
        # self._expand_vocab(self.words, self.vocab)

    @property
    def vocab_sorted(self):
        vlist = list(self.vocab)
        vlist.sort(key=lambda x: len(x))

    # def split(self, word):
    #     # replacements with largest first
    #     for
    #     pass

    def _expand_vocab(self, words, vocab):
        word_splits = [self.tokenize()]
        pass

    def _get_init_vocab(self, words):
        vocab = set()
        for w in words:
            for c in w:
                vocab.add(c)
        return vocab


if __name__ == "__main__":
    data = [["hello, my name is", "my name is what"]]
    tok = Tokenizer(data)

    print("Tokenizer Stats")
    print(f"    Vocab: {tok.vocab}")
