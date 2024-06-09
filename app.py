from flask import Flask, request, redirect, render_template, flash, session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import os
import easyocr
import numpy as np
import librosa
from pydub import AudioSegment, effects
import tensorflow as tf
import json
import sqlite3
import mail
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

with sqlite3.connect('users.db') as db:
    c = db.cursor()

c.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT,username TEXT UNIQUE NOT NULL,password TEXT NOT NULL,fullname TEXT NOT NULL,email TEXT NOT NULL,gemail TEXT NOT NULL);')
db.commit()
db.close()
model_rf = pickle.load(open('model_dt.pkl', 'rb'))
reader = easyocr.Reader(['en'])

app = Flask(__name__)
app.secret_key = 'top_secret_key' 

ALLOWED_EXTENSIONS = {'jpg','png'}
ALLOWED_EXTENSION = {'mp3','wav'}
app.config['UPLOAD_FOLDER'] = 'uploads'
def preprocess_audio(path):
    _, sr = librosa.load(path)
    raw_audio = AudioSegment.from_file(path)
    
    samples = np.array(raw_audio.get_array_of_samples(), dtype='float32')
    trimmed, _ = librosa.effects.trim(samples, top_db=25)
    padded = np.pad(trimmed, (0, 180000 - len(trimmed)), 'constant')
    return padded, sr

"""bot_data={"preventing ":"you can take the following steps \n.Education about the dangers of drug use.\nPromoting healthy, drug-free activities and environments.\nStrong parental involvement and monitoring.\nBuilding strong relationships and communication within the family.",
          "symptoms":"Following are the main symptoms\nChanges in physical appearance, including sudden weight loss or gain.\nWithdrawal from social activities.\nUnusual smells on breath, body, or clothing.\nChanges in behavior or personality.\nIncreased secrecy, lying, or stealing.",
          "community support":"Local health clinics often provide rehabilitation services or referrals.\nOrganizations like Alcoholics Anonymous (AA) or Narcotics Anonymous (NA).\nCommunity mental health centers.\nNon-profit organizations that focus on drug recovery.",
          "legal":"Consequences vary widely but can include fines, imprisonment, and mandatory rehabilitation.\nPossession, distribution, and manufacturing of illegal drugs are typically penalized.\nSome jurisdictions offer diversion programs for first-time offenders.",
          "mental health":"Drug abuse can exacerbate or trigger new instances of mental health disorders such as anxiety, depression, or schizophrenia. Substance use can impair cognitive functions and emotional stability. Withdrawal effects can also include severe mental health symptoms.",
          "preventing":"Implement comprehensive drug education programs.Enforce a strict no-drug policy on school premises. Train staff to recognize signs of drug use and provide appropriate interventions. Offer counseling and support services."

          }
f = open('bot.json')
data = json.load(f)
f.close()
q_list=[]
a_list=[]
for i in data['intents']:
    for j in i:
        if j=='tag':
            q_list.append(i[j])
            a_list.append(i['responses'][0])
        elif j=='patterns':
            for k in i[j]:
                q_list.append(k)
                a_list.append(i['responses'][0]) """

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

# Create your views here.

"""bot = ChatBot('chatbot',read_only=False,
                logic_adapters=[{
                    
                    
                    'import_path':'chatterbot.logic.BestMatch',
                    #'default_response':'Sorry I dont know what that means',
                    #'maximum_similarity_threshold':0.90
                    
                    
                    
                    }])

list_to_train =[

    "hi",
    "hi, there",
    "what's your name",
    "I am just a chatbot",
    "what is your fav food?",
    "I like cheese",
    "what's your fav sport?",
    "swimming",
    "do you have children?"
    "no",
    "do you love me?",
    "yes to the extent that I can't live without you"
]


chatterbotCorpusTrainer = ChatterBotCorpusTrainer(bot)
#list_trainer=ListTrainer(bot)
#list_trainer.train(list_to_train)

chatterbotCorpusTrainer.train('chatterbot.corpus.english')"""

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('bot.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
# training = np.array(training)
training = np.array(training, dtype=object)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")


from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('bot.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
def getResponse(ints, intents_json):
    #userMessage = request.GET.get('userMessage')
    #chatResponse = str(bot.get_response(userMessage))
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_file1(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION

@app.route('/')
@app.route('/login', methods=['POST','GET'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        c.execute('SELECT * FROM users WHERE username=?', (username,))
        user = c.fetchone()

        if not user:
            return render_template('login.html', error='Invalid username or password')

        if not check_password_hash(user[2], password):
            return render_template('login.html', error='Invalid username or password')

        session['username'] = username
        session['name'] = user[3]
        session['email'] = user[4]
        session['gemail'] = user[5]
        return redirect('/upload')

    return render_template('login.html')

@app.route('/register', methods=['POST','GET'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm-password']
        email = request.form['email']
        gemail = request.form['gemail']

        errors = []
        if len(name) < 3:
            errors.append("Name should be at least 3 characters long.")
        if len(username) < 5:
            errors.append("Username should be at least 5 characters long.")
        if len(password) < 6:
            errors.append("Password should be at least 6 characters long.")
        if password != confirm_password:
            errors.append("Passwords do not match.")
        
        if errors:
            return render_template('register.html', errors=errors)
        else:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()

            hashed_password = generate_password_hash(password)

            c.execute('SELECT * FROM users WHERE username=?', (username,))
            if c.fetchone():
                return render_template('register.html', error='Username already exists')

            c.execute('INSERT INTO users (username, password,fullname,gemail,email) VALUES (?,?,?,?,?)', (username, hashed_password,name,gemail,email))
            conn.commit()
            return redirect('/login')
    return render_template('register.html')


@app.route('/upload', methods=['GET', 'POST'])
def index():
    ref={'Phencyclidine (PCP) 25':25, 
     'Amphetamine':1000, 
     'CANNABINOIDS':50, 
     'Cocaine':300, 
     'Opiate':2000,
     'Amphetamine':1000,
     'Barbiturates':300,
 'Benzodiazepines':300,
 'Marijuana':10,
 'Cocaine':300,
 'Methaqualone':300,
 'Phencyclidine (PCP)':25,
 'Methadone':300,
 'Propoxyphene':300}
    if request.method == 'POST':
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
           
            result = reader.readtext(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flag=False
            result_lst=["positive","negative"]
            fflag=False
            lists_drug=[]
            lists_q=[]
            for i in result:
                if i[1]=="Cutoff":
                    flag=True
                elif i[1].split(" ")[0]=="Collected":
                    flag=False
                if flag and i[1]!="Cutoff":
                    if i[1].isdigit() and check_flag:
                        lists_q.append(i[1])
                        check_flag=False
                    elif i[1].isdigit():
                        pass
                    else:
                        lists_drug.append(i[1])
                        check_flag=True
            flag=False
            for i in lists_drug:
                if i in ref:
                    if int(lists_q[lists_drug.index(i)])>ref[i]:
                        flag=True
                        print(i)

            if flag:
                finalr=" Positive"
                mail.mail(session['gemail'],finalr)
            else:
                finalr=" Negative"
                                
            # else:
                
            #     finalr="No drug usage detected"
  
            return render_template('index.html', result=1,re=finalr)
    return render_template('index.html') 

@app.route('/quest', methods=['GET', 'POST'])
def quest():
    if request.method == 'POST':
        # Here you can handle the form input, like printing it or saving it to a database
        input1 = request.form.get('birth')
        # dropdown1 = request.form.get('region')
        input2 = request.form.get('sex')
        input4 = request.form.get('DAYSCOCAINE')
        input5 = request.form.get('MARYJDAYS')
        input6 = request.form.get('METHADONE')
        input7 = request.form.get('HALLUC')
        input8 = request.form.get('METHDAYS')
        input9 = request.form.get('ARRESTED')
        input10 = request.form.get('CRIMES')
        xin=np.array([input1,input2,input4,input5,input6,input7,input8,input9,input10])
        xin=np.expand_dims(xin,axis=0)
        #print(np.shape(input1,dropdown2,input3,input4,input5,input6,input7,input8,input9,input10,input11,input12))
        # xin = scaler.transform(xin)
        #xin=np.expand_dims(xin,axis=0)
        p=model_rf.predict(xin)
        print(p)
        print(type(p))
        if  p==0:
            str="No addiction detected"
        elif((p>0) and (p<=2)):
            str="Low level of addiction"
        elif(p>2 and p<=5):
            str="Moderate level of addiction"
        elif(p>5 and p<=8):
            str="Substantial level of addiction"
        else:
            str="Severe level of addiction"
        return render_template('result.html',prediction=p[0],string=str)
    return render_template('main.html')

@app.route('/voice', methods=['GET', 'POST'])
def voice():
    if request.method == 'POST':
        print(1)
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file1(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
           
            
            audio, _ = preprocess_audio(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            FRAME_LENGTH = 2048
            HOP_LENGTH = 512
            zcr = librosa.feature.zero_crossing_rate(audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
            rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
            mfccs = librosa.feature.mfcc(y=audio, sr=44100, n_mfcc=13, hop_length=HOP_LENGTH)
            X_features = np.concatenate((
                    np.swapaxes([zcr], 1, 2),
                    np.swapaxes([rms], 1, 2),
                    np.swapaxes([mfccs], 1, 2)),
                    axis=2
                )
            #x=np.array([zcr,rms,mfccs])
            model=tf.keras.models.load_model("voiceModel_1.h5")
            y=model.predict(X_features)
            p=np.argmax(y)
            print(p)
            if p==0:
                finalr="drug usage detected"
                mail.mail(session['gemail'],finalr)
                                
            else:
                finalr="No drug usage detected"
  
            #return process_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('voice.html', result=1,re=finalr)
    return render_template('voice.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    return render_template('cbot.html')

@app.route("/get")
def get_bot_response():
    response={}
    userText = str(request.args.get('msg'))
    #print(bot_data)
    ints = predict_class(userText, model)
    res = getResponse(ints, intents)
    return res
    """print(q_list)
    for i in q_list:
        if i in userText or userText in i:
            return a_list[q_list.index(i)] 
    return "Sorry could not understand" """




"""def chatbot_response(request):
    text=request.GET.get('userMessage')
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return HttpResponse(res)"""

if __name__ == '__main__':
    app.run(debug=True)