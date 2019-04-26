import nltk

examples = [
    "natural language sentence",
    "I am a natural language sentence"
]


def download_nltk_models():
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")


def turing(sentence):
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    print("POS: ", pos_tags)
    return len(sentence) / 100


def main():
    # download_nltk_models()
    run_turing(examples)


# region Helpers
def print_results(result_pairs):
    for (example, result) in result_pairs:
        print("%s -> %s" % (example, result))


def run_turing(collection):
    result_pairs = []
    for item in collection:
        result_pairs.append((item, turing(item)))
    print_results(result_pairs)


if __name__ == "__main__":
    main()
# endregion
