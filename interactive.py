from turing import load_classifier, turing_classify

SHOW_PROBA = False

easter_eggs = {
    "Senator, we run ads": "Zuckerberg",
    "Bite my shiny metal ass": "Bender"
}

prompts_turing = {
    "title": "NATURAL LANGUAGE DETECTOR",
    "challenge": "So you pretend you're a human? Type a query you would search on an e-commerce website:",
    "exit": "See you later meatbag!",
    "success": "You're a human! 🙋",
    "failure": "You sound like a robot 🤖",
    "nothing": "You did not enter anything. Definitely robot.",
}

prompts_classics = {
    "title": "Author detector",
    "challenge": "Let's see if you sound more like Shakespeare than Hobbes:",
    "exit": "Thou know'st 'tis common; all that lives must die,\nPassing through nature to eternity.",
    "success": "Sounds Shakespearian. 🎩",
    "failure": "Did you read too much Hobbes? 🤓",
    "nothing": "Analphabet.",
}

if __name__ == "__main__":
    classifier, used_classics = load_classifier()
    prompts = prompts_classics if used_classics else prompts_turing
    print("\n\n" + prompts["title"] + "\n")

while True:
    word = input("\n" + prompts["challenge"] + "\n")
    if word == "/exit":
        print(prompts["exit"])
        break
    elif word in easter_eggs:
        print("You sound like %s. Definitely a robot." % easter_eggs[word])
        continue
    elif len(word) == 0:
        print("You did not enter anything. Definitely robot.")
    is_human, proba = turing_classify(classifier, word)
    proba_string = " ({0:.0%} sure)".format(proba) if SHOW_PROBA else ""
    print((prompts["success" if is_human else "failure"]) + proba_string)
    continue
