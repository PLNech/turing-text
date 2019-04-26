import nltk
from data import *

from textblob.classifiers import NaiveBayesClassifier


def download_nltk_models():
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")


def get_training_data():
    tuples = []
    for example in train_positives:
        # tokens = nltk.word_tokenize(example) TODO Use instead of raw sentnece
        tuples.append((example, True))
    for example in train_negatives:
        tuples.append((example, False))
    return tuples


def train_classifier():
    return NaiveBayesClassifier(get_training_data())


def turing_classify(sentence):
    classifier = train_classifier()
    classify = classifier.classify(sentence)
    print("Classifier(%s) -> %s" % (sentence, classify))
    return classify


def main():
    # download_nltk_models()
    print("Negatives:")
    run_turing(test_negatives, False)
    print("\n\n\nPositives:")
    run_turing(test_positives, True)


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
