import random
import pickle
import nltk
from data import *

from textblob.classifiers import NaiveBayesClassifier

DEBUG = False


def train_and_test():
    # download_nltk_models()
    train_positives, train_negatives, test_positives, test_negatives = build_datasets()
    training_data = train_positives + train_negatives
    random.shuffle(training_data)
    debug("Training data", training_data)

    classifier = train_classifier(training_data)

    accuracy = run_turing(classifier, [data for (data, label) in test_negatives], False)
    print("Accuracy for negative examples: %s" % accuracy)

    accuracy = run_turing(classifier, [data for (data, label) in test_positives], True)
    print("Accuracy for positive examples: %s" % accuracy)

    save_classifier(classifier)


def build_datasets():
    random.shuffle(positives)
    random.shuffle(negatives)
    positives_pos = [get_pos_tags(it) for it in positives]
    negatives_pos = [get_pos_tags(it) for it in negatives]
    part_pos = partition(positives_pos, 5)
    part_neg = partition(negatives_pos, 5)

    train_positives = make_tuples(part_pos[0] + part_pos[1] + part_pos[2] + part_pos[3], True)
    train_negatives = make_tuples(part_neg[0] + part_neg[1] + part_neg[2] + part_neg[3], False)
    test_positives = make_tuples(part_pos[4], True)
    test_negatives = make_tuples(part_neg[4], False)
    debug("Generated data!\nTrain+:%s\nTrain-:%s\nTest+:%s\nTest-:%s\n" % (
        train_positives, train_negatives, test_positives, test_negatives))
    return train_positives, train_negatives, test_positives, test_negatives


def make_tuples(data, value):
    return [(it, value) for it in data]


def get_pos_tags(example):
    tokens = nltk.word_tokenize(example)
    pos_tags = nltk.pos_tag(tokens)
    tags = [v for (k, v) in pos_tags]
    return tags


def train_classifier(data):
    debug("Training classifier on data like", data[0])
    classifier = NaiveBayesClassifier(data)
    debug("Done training!")
    return classifier


def print_results(result_pairs):
    for (example, result) in result_pairs:
        print("%s -> %s" % (example, result))


def save_classifier(classifier):
    f = open('classifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()


def load_classifier():
    f = open('classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier


# region Helpers
def main():
    train_and_test()


def run_turing(classifier, collection, expected):
    result_pairs = []
    for input in collection:
        result_pairs.append((input, turing_classify_pos(classifier, input)))
    # print_results(result_pairs)
    accuracy = sum([1 for (item, output) in result_pairs if output is expected]) / len(result_pairs)
    return accuracy


def turing_classify_pos(classifier, sentence):
    classify = classifier.prob_classify(sentence)
    prediction = classify.max()
    probability = classify.prob(prediction)
    debug("Classifier(%s) -> %s (%s)" % (sentence, prediction, probability))
    return prediction, probability


def turing_classify(classifier, sentence):
    return turing_classify_pos(classifier, get_pos_tags(sentence))


def partition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]


def download_nltk_models():
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")


def debug(text, *texts):
    if DEBUG:
        print(text, *texts)


if __name__ == "__main__":
    main()
# endregion
