example = "natural language sentence"
example2 = "I am a natural language sentence"


def turing(string):
    proba = len(string) / 100
    return proba


def main():
    print("Let's run turing on %s: %s" % (example, turing(example)))


if __name__ == "__main__":
    main()
