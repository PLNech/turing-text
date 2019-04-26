examples = [
    "natural language sentence",
    "I am a natural language sentence"
]


def print_results(result_pairs):
    for (example, result) in result_pairs:
        print("%s -> %s" % (example, result))


def run_turing(collection):
    result_pairs = []
    for item in collection:
        result_pairs.append((item, turing(item)))
    print_results(result_pairs)


def turing(string):
    proba = len(string) / 100
    return proba


def main():
    run_turing(examples)


if __name__ == "__main__":
    main()
