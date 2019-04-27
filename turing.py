import random
import pickle
import nltk
from data import positives as data_positives
from data import negatives as data_negatives
from data_classics import positives as classics_positives
from data_classics import negatives as classics_negatives

from textblob.classifiers import NaiveBayesClassifier

DEBUG = True
USE_CLASSICS = True


def name_feature(is_positive):
    if is_positive:
        return "Shakespeare" if USE_CLASSICS else "positive"
    else:
        return "Hobbes" if USE_CLASSICS else "negative"


def build_datasets():
    positives = classics_positives if USE_CLASSICS else data_positives
    negatives = classics_negatives if USE_CLASSICS else data_negatives
    random.shuffle(positives)
    random.shuffle(negatives)
    min_size = min(len(positives), len(negatives))
    debug("Dataset size: len(pos)=%s, len(neg)=%s -> %s" % ((len(positives)), (len(negatives)), min_size))

    positives_pos = [get_pos_tags(it) for it in positives[0:min_size]]
    negatives_pos = [get_pos_tags(it) for it in negatives[0:min_size]]

    example_positive = "Example " + name_feature(True) + " data: %s -> %s"
    example_negative = "Example " + name_feature(False) + " data: %s -> %s"
    print(example_positive % (positives[0], positives_pos[0]))
    print(example_negative % (negatives[0], negatives_pos[0]))

    part_pos = partition(positives_pos, 5)
    part_neg = partition(negatives_pos, 5)

    train_positives = make_tuples(part_pos[0] + part_pos[1] + part_pos[2] + part_pos[3], True) # TODO: LIST COMPrehension
    train_negatives = make_tuples(part_neg[0] + part_neg[1] + part_neg[2] + part_neg[3], False)
    test_positives = make_tuples(part_pos[4], True)
    test_negatives = make_tuples(part_neg[4], False)
    debug("Generated data!\nTrain+:%s\nTrain-:%s\nTest+:%s\nTest-:%s\n" % (
        train_positives, train_negatives, test_positives, test_negatives))
    return train_positives, train_negatives, test_positives, test_negatives


def train_and_test():
    download_nltk_models()
    train_positives, train_negatives, test_positives, test_negatives = build_datasets()
    training_data = train_positives + train_negatives
    random.shuffle(training_data)
    debug("Training data", training_data)

    classifier = train_classifier(training_data)

    accuracy = run_turing(classifier, [data for (data, label) in test_negatives], False)
    print("Accuracy for " + name_feature(False) + " examples: %s" % accuracy)

    accuracy = run_turing(classifier, [data for (data, label) in test_positives], True)
    print("Accuracy for " + name_feature(False) + " examples: %s" % accuracy)

    print("Classifier features:", classifier.show_informative_features(5))
    save_classifier(classifier)


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
    pickle.dump(USE_CLASSICS, f)
    pickle.dump(classifier, f)
    f.close()


def load_classifier():
    f = open('classifier.pickle', 'rb')
    used_classics = pickle.load(f)
    classifier = pickle.load(f)
    f.close()
    return classifier, used_classics


# region Helpers
def main():
    train_and_test()


def run_turing(classifier, collection, expected):
    result_pairs = []
    avg_proba = 0
    for sentence in collection:
        (prediction, probability) = turing_classify_pos(classifier, sentence)
        avg_proba += probability
        result_pairs.append((sentence, prediction, probability))
    avg_proba /= len(collection)
    debug("example result:", result_pairs[0])
    print("Average probability: %s" % avg_proba)
    return sum([1 for (item, prediction, probability) in result_pairs if prediction is expected]) / len(result_pairs)


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
