import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import os
import random
from sklearn.ensemble import RandomForestClassifier
import pickle

# Set the stop words in English language
stopset = set(stopwords.words('english'))
# Download the Tagger library
nltk.download('averaged_perceptron_tagger')
# Lemmatize the words -- i.e. bring back the words to basic form
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Small talk inputs and outputs
GREETING_INPUTS = ["hello", "hi", "greetings", "sup", "what's up", "hey"]
GREETING_RESPONSES = ["Hi there😇", "*nods*"]

def train_model(tagger, entity_list, pos_tag, word_occurance_position, total_set_of_words):
    print(tagger, entity_list, word_occurance_position)
    print("\n###################### Parsing the inputs ######################\n")
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
    df.to_csv("Training.csv", index = False)
    print("Saved to csv!")

    print("###################### Pre-processing the training file ######################\n")
    pos_tag_one_hot = pd.get_dummies(df["pos_tag"])
    df = pd.concat([df, pos_tag_one_hot], axis = 1)
    df.drop(["pos_tag"], axis = 1, inplace = True) ## Finished the One-Hot encoding of the file
    
    ## Train the model using the df as input
    x_cols = [x for x in df.columns if x != "position" and x != "word"]
    rfc = RandomForestClassifier(n_jobs = -1, random_state = 42, n_estimators = 100)
    rfc.fit(df[x_cols], df["position"])
    print(rfc.feature_importances_)
    print(rfc.predict([[3, 0, 0, 0, 0, 1]]))
    print(rfc.predict([[3, 0, 1, 0, 0, 0]]))
    print(rfc.predict([[7, 0, 1, 0, 0, 0]]))

    ## Store the model for the prediction used-case
    filename = "Training_Model.pkl"
    pickle.dump(rfc, open(filename, 'wb'))

    df.to_csv("Training2.csv", index = False)

if input("Want to Enter the training phase or the Testing Phase? ") == "train":
    entity_list = []; pos_tag = []; word_occurance_position = []; total_set_of_words = []
    while True:
        try:
            ## I have to get information about the position of the entities
            ## What are thier POS Tags
            ## -- wrt after lemm and stop removal
            intents = str(input("Me: ")).lower()
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
    while True:
        try:
            ## Code for prediction from the trained model
            input_intent = str(input("Me: ")).lower()
            tokenize = nltk.word_tokenize(input_intent) ## Tokenize the sentence
            ## Remove the stop words and lemmatize the words
            tokens = [w for w in tokenize if not w in stopset]
            tagger = nltk.pos_tag(tokens) ## Get the POS tags as well
            for items in tagger:
                if items in GREETING_INPUTS:
                    print("Chatbot: ", random.choice(GREETING_RESPONSES))
                if items == "?":
                    print("Chatbot: I am IBOT. Your chatbot companion!")
                if 
        except (KeyboardInterrupt):
            break