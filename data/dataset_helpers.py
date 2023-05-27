from hyper_parameters import *


def word_to_index_dict(tagged_file):
    special_words = [NO_WORD]
    distinct_words = set()
    suffixes = set()
    prefixes = set()
    characters = set()
    with open(tagged_file) as f:
        for line in f:
            if line == "\n" or 'DOCSTART' in line:
                continue
            try:
                word, _ = line.rstrip().split()
                word = word.lower()
            except Exception:
                print(line)
                input()

            distinct_words.add(word)
            prefixes.add(word[:3])
            suffixes.add(word[-3:])
            characters.update(word)
    word_to_index = {word: index + len(special_words) for index, word in enumerate(distinct_words)}
    prefix_to_index = {word: index + len(special_words) for index, word in enumerate(prefixes)}
    suffix_to_index = {word: index + len(special_words) for index, word in enumerate(suffixes)}
    characters = {word: index + len(special_words) for index, word in enumerate(characters)}
    for offset, word in enumerate(special_words):
        word_to_index[word] = offset
        prefix_to_index[word] = offset
        suffix_to_index[word] = offset
        characters[word] = offset

    return word_to_index, prefix_to_index, suffix_to_index, characters


def generate_texts_labels(tagged_file, word_to_index, prefix_to_index, suffix_to_index, char_to_index, tag_to_index,
                          dont_include, tagged):
    text_buffer = []
    suf_buf = []
    pre_buf = []
    char_buf = []
    label_buffer = []

    with open(tagged_file) as f:

        for line in f:

            if 'DOCSTART' in line:
                continue

            # A sentence is ended
            if line == "\n":
                if tagged:
                    yield text_buffer, pre_buf, suf_buf, char_buf, label_buffer
                else:
                    yield text_buffer, pre_buf, suf_buf, char_buf
                text_buffer, suf_buf, pre_buf, char_buf, label_buffer = [], [], [], [], []
                continue

            if tagged:
                word, tag = line.rstrip().split()

                if not tag in tag_to_index:
                    dont_include.append(tag)
                    continue
            else:
                word = line.rstrip()
            word = word.lower()

            text_buffer.append(word_to_index.get(word) or word_to_index[NO_WORD])
            pre_buf.append(prefix_to_index.get(word[:3]) or prefix_to_index[NO_WORD])
            suf_buf.append(suffix_to_index.get(word[-3:]) or suffix_to_index[NO_WORD])

            chars = [char_to_index.get(c) or char_to_index[NO_WORD] for c in word]

            char_buf.append(chars)

            if tagged:
                label_buffer.append(tag_to_index[tag])
