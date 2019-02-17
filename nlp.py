import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import os

# Set the stop words in English language
stopset = set(stopwords.words('english'))
# Download the Tagger library
nltk.download('averaged_perceptron_tagger')
# Lemmatize the words -- i.e. bring back the words to basic form
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def train_model(tagger, entity_list, pos_tag, word_occurance_position, total_set_of_words):
    print(tagger, entity_list, word_occurance_position)
    exists = os.path.isfile('Training.csv')
    if not exists:
        df = pd.DataFrame(columns = ["word", "pos_tag", "position", "total_w"])
        i = 0
    else:
        df =  pd.read_csv("Training.csv")
        i = df.shape[0]
    for ent, po, wo, tsw in zip(entity_list, pos_tag, word_occurance_position, total_set_of_words):
        df.at[i, "word"] = ent
        df.at[i, "pos_tag"] = po
        df.at[i, "position"] = wo
        df.at[i, "total_w"] = tsw
        i = i + 1
    print("\n###################### Completed expanding! ######################\n")
    df.to_csv("Training.csv", index = False)
    print("Saved to csv!")

if input("Want to Enter the training phase or the Testing Phase? ") == "train":
    entity_list = []; pos_tag = []; word_occurance_position = []; total_set_of_words = []
    while True:
        try:
            ## I have to get information about the position of the entities
            ## What are thier POS Tags
            ## -- wrt after lemm and stop removal
            intents = str(input("Me: "))
            tokenize = nltk.word_tokenize(intents) ## Tokenize the sentence
            ## Remove the stop words and lemmatize the words
            tokens = [lemmatizer.lemmatize(w) for w in tokenize if not w in stopset]
            tagger = nltk.pos_tag(tokens) ## Get the Part-of-Speech Tagger info
            for index, items in enumerate(tagger):
                print(items[0], " <-- a Entity? y for YES")
                if input() == "y":
                    entity_list.append(items[0])
                    pos_tag.append(items[1])
                    word_occurance_position.append(index)
                    total_set_of_words.append(len(tagger))    

            print(tagger, entity_list, word_occurance_position, pos_tag, total_set_of_words)
        except (KeyboardInterrupt):
            if input("\nDo you want to finish entering input to training? ") == "n":
                ## Continue to gather intents and entities
                continue
            else:
                ## Code for training based on the information gathered
                print("\n###################### Now -- Training! ######################\n")
                train_model(tagger, entity_list, pos_tag, word_occurance_position, total_set_of_words)
                break
else:
    ## Code for prediction from the trained model
    pass