test_positives = [
    "I'm looking for a movie with Tom Cruse",
    "I need a recent car with four doors",
    "When was Algolia founded?",
    "I'm in the mood for an action movie",
    "A blue dress with long sleeves",
]

train_positives = [
    "I want some red shoes for a wedding",
    "I want to book a trip tonight for vegas",
    "An iPhone with 32gb",
    "Who's working as an SMB in Paris?",
    "Get me to the closest bakery",
    "A book about brain science",
    "A recent book on philosophy",
    "I want to read a love novel",
    "I ate a banana and chocolate",
    "I want an athletic male in his thirties that loves hiking",
    "Is there a method to classify sentences in natural language?",
    "This subway is crowded, this is getting painful!",
    "I'm getting there, but we are not done.",
    "How can I index blog posts?",
    "Is there a limit to the size of records?",
    "Do you have a search NLU expert available?",
    "What's the core idea of search indexing?",
    "An analysis of coworking spaces in Paris",
]

test_negatives = [
    "IPhone 32gb",
    "Athletic male thirties hiking",
    "Algolia foundation date",
    "Index blog posts",
    "Closest bakery",
]

train_negatives = [
    "Red wedding shoes",
    "Action movie",
    "Vegas tonight",
    "SMB Paris",
    "Tom Cruse",
    "Blue dress long sleeves",
    "Brain science",
    "Recent philosophy",
    "Love novel",
    "Banana and chocolate",
    "Classify natural language sentence",
    "Recent four doors car",
    "Crowded subway painful",
    "Getting there but not done",
    "Record size limit",
    "Available search NLU expert",
    "Search indexing core idea",
    "Paris coworking space analysis",
]

train_all = train_negatives + train_positives
test_all = test_negatives + test_positives
