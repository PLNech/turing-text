import nltk
from examples import *

from textblob.classifiers import NaiveBayesClassifier

def download_nltk_models():
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")


def turing_classify(sentence):
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    # print("POS: ", pos_tags)
    score = len(sentence) / 100
    return score >= 0.25


def main():
    # download_nltk_models()
    print("Negatives:")
    run_turing(examples_negatives, False)
    print("\n\n\nPositives:")
    run_turing(examples_positives, True)


# region Helpers
def print_results(result_pairs):
    for (example, result) in result_pairs:
        print("%s -> %s" % (example, result))


def run_turing(collection, expected):
    result_pairs = []
    for item in collection:
        result_pairs.append((item, turing_classify(item)))
    # print_results(result_pairs)
    accuracy = sum([1 for (item, output) in result_pairs if output is expected]) / len(result_pairs)
    print("Accuracy: %s" % accuracy)
    return accuracy


if __name__ == "__main__":
    main()
# endregion
