from hyper_parameters import *


def word_to_index_dict(tagged_file, special_words=None):
    if special_words is None:
        special_words = [START_PAD, END_PAD, NO_WORD]

    distinct_words = set()
    with open(tagged_file) as f:
        for line in f:
            if line == "\n":
                continue
            word, _ = line.rstrip().split("\t")
            distinct_words.add(word.lower())
    word_to_index = {word: index for index, word in enumerate(distinct_words)}
    for offset, word in enumerate(special_words):
        word_to_index[word] = len(distinct_words) + offset

    return word_to_index


def generate_texts_labels(tagged_file, word_to_index, tag_to_index, prob_replace_to_no_word=PROB_UNQ):

    start_embedding = word_to_index[START_PAD]
    end_embedding = word_to_index[END_PAD]
    start_buffer = [start_embedding, start_embedding, end_embedding, end_embedding]
    text_buffer = start_buffer.copy()
    label_buffer = []
    start_index = WINDOW // 2 + (1 - (WINDOW % 2))
    current_index = start_index

    with open(tagged_file) as f:

        for line in f:

            # A sentence is ended
            if line == "\n":
                for i in range(len(text_buffer) - (WINDOW - 1)):
                    # Convert a list of embedding of 5 words to a single vector
                    yield text_buffer[i:i + WINDOW], label_buffer[i]
                text_buffer = start_buffer.copy()
                label_buffer = []
                current_index = start_index
                continue

            word, tag = line.rstrip().split("\t")
            if torch.rand(1) < prob_replace_to_no_word:
                word = NO_WORD
            embedding = word_to_index.get(word.lower()) or word_to_index[NO_WORD]
            text_buffer.insert(current_index, embedding)
            current_index += 1
            label_buffer.append(tag_to_index[tag])
