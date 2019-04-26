from turing import load_classifier, turing_classify

SHOW_PROBA = False

easter_eggs = {
    "Senator, we run ads": "Zuckerberg",
    "Bite my shiny metal ass": "Bender"
}

if __name__ == "__main__":
    classifier = load_classifier()
while True:
    word = input("So you pretend you're a human? Type a sentence to prove it! (or /exit to quit)\n")
    if word == "/exit":
        print("See you later meatbag!")
        break
    elif word in easter_eggs:
        print("You sound like %s. Definitely a robot." % easter_eggs[word])
        continue
    elif len(word) == 0:
        print("You did not enter anything. Definitely robot.")
    is_human, proba = turing_classify(classifier, word)
    proba_string = " ({0:.0%} sure)".format(proba) if SHOW_PROBA else ""
    print(("You're a human!" if is_human else "YO FELLOW ROBOT") + proba_string)
    continue
