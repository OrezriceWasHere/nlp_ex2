from hyper_parameters import *


def word_to_index_dict(tagged_file):
    special_words = [START_PAD, END_PAD, NO_WORD]
    distinct_words = set()
    suffixes = set()
    prefixes = set()
    characters = set()
    with open(tagged_file) as f:
        for line in f:
            if line == "\n":
                continue
            try:
                word, _ = line.rstrip().split()
                word = word.lower()
                word = word[:MAX_CHARACTERS]
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
                          dont_include, tagged, presuf=True, with_chars=True):
    start_embedding = word_to_index[START_PAD]
    end_embedding = word_to_index[END_PAD]
    start_buffer = [start_embedding, start_embedding, end_embedding, end_embedding]
    char_start_buffer = [[start_embedding] * MAX_CHARACTERS, [start_embedding] * MAX_CHARACTERS,
                         [end_embedding] * MAX_CHARACTERS, [end_embedding] * MAX_CHARACTERS]
    text_buffer = start_buffer.copy()
    suf_buf = start_buffer.copy()
    pre_buf = start_buffer.copy()
    char_buf = char_start_buffer.copy()
    label_buffer = []
    start_index = WINDOW // 2 + (1 - (WINDOW % 2))
    current_index = start_index

    with open(tagged_file) as f:

        for line in f:

            if 'DOCSTART' in line:
                continue

            # A sentence is ended
            if line == "\n":
                for i in range(len(text_buffer) - (WINDOW - 1)):
                    # Convert a list of embedding of 5 words to a single vector
                    if tagged:
                        yield text_buffer[i:i + WINDOW], pre_buf[i:i + WINDOW], suf_buf[i:i + WINDOW], \
                              char_buf[i: i + WINDOW], label_buffer[i]
                    else:
                        if presuf:
                            yield text_buffer[i:i + WINDOW], pre_buf, suf_buf
                        elif with_chars:
                            yield text_buffer[i:i + WINDOW], char_buf[i: i + WINDOW]
                        else:
                            yield text_buffer[i:i + WINDOW]
                text_buffer, suf_buf, pre_buf = start_buffer.copy(), start_buffer.copy(), start_buffer.copy()
                char_buf = char_start_buffer.copy()
                label_buffer = []
                current_index = start_index
                continue

            if tagged:
                word, tag = line.rstrip().split()

                if not tag in tag_to_index:
                    dont_include.append(tag)
                    continue
            else:
                word = line.rstrip()
            word = word.lower()
            word = word[:MAX_CHARACTERS]

            text_buffer.insert(current_index, word_to_index.get(word) or word_to_index[NO_WORD])
            pre_buf.insert(current_index, prefix_to_index.get(word[:3]) or prefix_to_index[NO_WORD])
            suf_buf.insert(current_index, suffix_to_index.get(word[-3:]) or suffix_to_index[NO_WORD])

            chars = [word_to_index[NO_WORD]] * MAX_CHARACTERS
            for i, c in enumerate(word):
                try:
                    chars[i + (MAX_CHARACTERS - len(word)) // 2] = char_to_index.get(c) or char_to_index[NO_WORD]
                except IndexError:
                    print(word)
                    input()

            char_buf.insert(current_index, chars)

            current_index += 1
            if tagged:
                label_buffer.append(tag_to_index[tag])
