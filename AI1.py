import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()  
#nltk.download('punkt')

import tensorflow as tf
import numpy as np
import tflearn 
import json
import random
import pickle

PathIntents = "/Users/marioharoun/Downloads/json file/intents.json"

##DATASORTERING

with open(PathIntents) as file:
    data = json.load(file)

try: 
    with open("data.pickle", "rb") as f: # "rb" = read as bytes
        ordene, klasser, trening, output = pickle.load(f)
except:
    ordene = []
    klasser = []
    dokumenter_pat = []
    dokumenter_tag = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            ordd = nltk.word_tokenize(pattern) # hva gjør tonkenize?? 
            ordene.extend(ordd)
            dokumenter_pat.append(ordd)
            dokumenter_tag.append(intent["tag"])

        if intent["tag"] not in klasser:
            klasser.append(intent["tag"])


    ordene = [stemmer.stem(w.lower()) for w in ordene if w != "?"]
    ordene = sorted(list(set(ordene)))
    klasser = sorted(klasser)

    trening = []  #skal inneholde en haug med lister av 0 og 1 som AI-en skjønner (one hot encoded)
    output = []   # ---||----

    out_tom = [0 for i in range(len(klasser))]   

    for x, doc in enumerate(dokumenter_pat):
        bag_Of_Words = [] 
        ordd = [stemmer.stem(w) for w in doc]  # finner stammen av ordene

        for i in ordene:
            if i in ordd:
                bag_Of_Words.append(1)
            else:
                bag_Of_Words.append(0)

        output_rad = out_tom[:]
        output_rad[klasser.index(dokumenter_tag[x])] = 1   #sjkker hvor i klassser tag-en er og sett inn 1 for den verdien

        trening.append(bag_Of_Words)
        output.append(output_rad)

    trening = np.array(trening)
    output = np.array(output)

    with open("data.pickle", "wb") as f: 
        pickle.dump((ordene, klasser, trening, output), f)

##TRENING AI-en  

tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(trening[0])]) #input-formen vi forventer aksepteres av modellen
net = tflearn.fully_connected(net, 8)  #har 8 neural for en "hidden-layer"
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #output layer, activation gir sannsynlighet for hver neuron in layer-en som vil være output for nettverket
net = tflearn.regression(net)

modell = tflearn.DNN(net)

"""try: 
    modell.load("modell.tflearn")
except:"""  # fikk problemer med denne og måtte fjerne den, men nå trenes modelle hver gang...

modell.fit(trening, output, n_epoch=3000, batch_size=8, show_metric=True) #n_epoch er antall ganger vi viser daten til modellen 
modell.save("modell.tflearn")

####

def bag_ord(s, ordene):
    bag_words = [0 for i in range(len(ordene))] #bare en liste med mange 0-er, for så å bytte ut 0 med en der ord eksis.
    
    s_ordene = nltk.word_tokenize(s)
    s_ordene = [stemmer.stem(w.lower()) for w in s_ordene]

    for se in s_ordene:
        for i, w in enumerate(ordene):
            if w == se:
                bag_words[i] = 1

    bag = np.array(bag_words)
    return bag

def chat():
    print("How can I help you? (type 'Quit' to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        resultater = modell.predict([bag_ord(inp, ordene)])[0]
        resultater_index = np.argmax(resultater) #gir indeks av største verdi i liste: høyest sannsynlighet
        tag = klasser[resultater_index]  #gir hvilken tag modellen tror man skal svare med

        #if resultater[resultater_index] > 0.7:
        for t in data["intents"]:
            if t["tag"] == tag:
                respons = t["responses"]

        print(random.choice(respons))
        """else:
            print("I don't understand. Try another question, please")"""

chat()