# Flask constructor takes the name of
# current module (_name_) as argument.

from flask import Flask, render_template, request, jsonify,url_for,redirect
import pickle
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
# import pickle

# open the PKL file and load the Python object


app = Flask(__name__, template_folder='templates', static_folder='static')

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.


@app.route('/', methods=["GET", "POST"])
# ‘/’ URL is bound with hello_world() function.
def home():         
    if request.method == "POST":
        # print(123)
        text = request.form['txt']
        print(text)
        model = request.form['model']
        print(model)
        df = text # assume input data is in JSON format
        if model=='bert' :
           run_notebook_bert(df)
           with open('model.pkl', 'rb') as f:
            data = pickle.load(f)
           return render_template("result.html",data=data) 
        else:
            run_notebook_codebert(df)
            with open('model1.pkl', 'rb') as f:
             data = pickle.load(f)
            return render_template("result.html",data=data) 
                
            
             
        #output = run_notebook(input_data)
        #print(output)
        #return jsonify({'output': output})
    return render_template('index.html')

# @app.route('/run_my_notebook', methods=['POST'])
# def run_my_notebook():
#     input_data = txt # assume input data is in JSON format
#     output = run_notebook(input_data)
#     return jsonify({'output': output})


def run_notebook_bert(df):

    import requests
 
    # get URL

    # url = input("Enter url: ")
    # page = requests.get(url)
    
    # # display status code
    # print(page.status_code)
    
    # # display scraped data
    # print(page.content)
    # import required modules
    from bs4 import BeautifulSoup
    # import requests
    
    # get URL
    #url = input("Enter url: ")
    # page = requests.get(url)
    
    # scrape webpage
    # soup = BeautifulSoup(page.content, 'html.parser')
    
    # display scraped data
    # html_text = soup.prettify()
    # print(html_text)
    import re
    # re_html = re.compile(r'<[^>]+>')
    # re_html.sub('',html_text)
    import pandas as pd
    import string
    # df=input('give string')
    string.punctuation
    def remove_punctuations(text):
        punctuations = string.punctuation
        return text.translate(str.maketrans('', '', punctuations))
    df = remove_punctuations(df)
    #df
    ### Removal of stopwords

    #Stop word removal is one of the most commonly used preprocessing steps across different NLP applications. The idea is simply removing the words that occur commonly across all the documents in the corpus. 
    import nltk
    from nltk.corpus import stopwords

    nltk.download('stopwords')
    sample_stopwords = stopwords.words('english')
    print(sample_stopwords)
    def remove_stopwords(text):
        return " ".join([word for word in text.split() if word not in sample_stopwords])
    df = remove_stopwords(df)
    #df
    ### Removal of frequent words
    from collections import Counter

    word_count = Counter()
    for text in df:
        for word in text.split():
            word_count[word] += 1
            
    word_count.most_common(10)
            
    frequent_words = set(word for (word,wc) in word_count.most_common(3))
    def remove_freq_words(text):
        
        return " ".join([word for word in text.split() if word not in frequent_words])
    df = remove_freq_words(df)
    #df
    ### Removal of Rare words
    rare_words = set(word for (word,wc) in word_count.most_common()[:-10:-1])
    rare_words
    def remove_rare_words(text):
        
        return " ".join([word for word in text.split() if word not in rare_words])
    df = remove_rare_words(df)
    #df
    ### Removal of Special Characters
    import re

    def remove_spl_characters(text):
        
        text = re.sub('[^a-zA-Z0-9]', ' ', text)
        text = re.sub('\s+', ' ',text)
        
        return text
    df = remove_spl_characters(df)
    #df
    ### Stemming

    #Stemming is the process of reducing a word to its stem that affixes to suffixes and prefixes or to the roots of words known as "lemmas".
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    def stemming(text):
        return " ".join([stemmer.stem(word) for word in text.split()])
            
    df = stemming(df)
    #df
    ### Lemmatization

    #Lemmatization is a text normalization technique used in Natural Language Processing (NLP), that switches any kind of a word to its base root mode. Lemmatization is responsible for grouping different inflected forms of words into the root form, having the same meaning.
    import nltk
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    from nltk import pos_tag
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()

    wordnet_map = {"N": wordnet.NOUN , "V" : wordnet.VERB , "J" : wordnet.ADJ , "A" : wordnet.ADV}

    def lemmatize(text):
        pos_text = pos_tag(text.split())
        return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word,pos in pos_text])
    df = lemmatize(df)
    #df
    ## Loading the pretrained model BERT(Bidirectional Encoder Representations from Transformers)

    # About bert-base-multilingual-uncased-sentiment:

    # This a bert-base-multilingual-uncased model finetuned for sentiment analysis on product reviews in six languages: English, Dutch, German, French, Spanish and Italian. It predicts the sentiment of the review as a number of stars (between 1 and 5).

    # This model is intended for direct use as a sentiment analysis model for product reviews in any of the six languages above, or for further finetuning on related sentiment analysis tasks.

    # !pip install transformers


    from transformers import AutoTokenizer , AutoModelForSequenceClassification
    import torch


    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

    tokens = tokenizer.encode(df, return_tensors='pt',truncation = True)


    result = model(tokens)

    result.logits

    int(torch.argmax(result.logits))+1

    # loading library
    # import picklecreate an iterator object with write permission - model.pkl
    # with open('model_pkl', 'wb') as files:
    #     pickle.dump(model, files)
    #     model=pickle.load(open('model_pkl', 'rb'))
    # loading dependency
    #import joblib
    #To save the model we will use its dump functionality to save the model to the model_jlib file.

    # saving our model # model - model , filename-model_jlib
    #joblib.dump(model , 'model_jlib')
    # loading library
    import pickle
    # create an iterator object with write permission - model.pkl
    with open('model.pkl', 'wb') as files:
        pickle.dump(model, files)
        #bert ends
        # import pickle

    # open the PKL file and load the Python object
    # with open('model1.pkl', 'rb') as f:
    #     data = pickle.load(f)

    # print the Python object
 #print(data)


    # print('hi')
    # # Read the ipynb file
    # with open('NLP_Automation_bert.ipynb') as f:
    #     nb = nbformat.read(f, as_version=4)
    # # Execute the code in the file with the input data
    # ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    # ep.preprocess(nb, {'metadata': {'path': '.'}, 'input': input_data})
    # # Get the output of the code
    # #output = nbformat.writes(nb)
    # model_pkl= nbformat.writes(nb)
    

    
    

def run_notebook_codebert(df):
    import requests
 
    # get URL

    # url = input("Enter url: ")
    # page = requests.get(url)
    
    # display status code
    # print(page.status_code)
    
    # display scraped data
    # print(page.content)
    # import required modules
    from bs4 import BeautifulSoup
    # import requests
    
    # get URL
    #url = input("Enter url: ")
    # page = requests.get(url)
    
    # # scrape webpage
    # soup = BeautifulSoup(page.content, 'html.parser')
    
    # # display scraped data
    # html_text = soup.prettify()
    # print(html_text)
    # import re
    # re_html = re.compile(r'<[^>]+>')
    # re_html.sub('',html_text)
    import pandas as pd
    import string
    # df=input('give string')
    string.punctuation
    def remove_punctuations(text):
        punctuations = string.punctuation
        return text.translate(str.maketrans('', '', punctuations))
    df = remove_punctuations(df)
    #df
    ### Removal of stopwords

    #Stop word removal is one of the most commonly used preprocessing steps across different NLP applications. The idea is simply removing the words that occur commonly across all the documents in the corpus. 
    import nltk
    from nltk.corpus import stopwords

    nltk.download('stopwords')
    sample_stopwords = stopwords.words('english')
    print(sample_stopwords)
    def remove_stopwords(text):
        return " ".join([word for word in text.split() if word not in sample_stopwords])
    df = remove_stopwords(df)
    #df
    ### Removal of frequent words
    from collections import Counter

    word_count = Counter()
    for text in df:
        for word in text.split():
            word_count[word] += 1
            
    word_count.most_common(10)
            
    frequent_words = set(word for (word,wc) in word_count.most_common(3))
    def remove_freq_words(text):
        
        return " ".join([word for word in text.split() if word not in frequent_words])
    df = remove_freq_words(df)
    #df
    ### Removal of Rare words
    rare_words = set(word for (word,wc) in word_count.most_common()[:-10:-1])
    rare_words
    def remove_rare_words(text):
        
        return " ".join([word for word in text.split() if word not in rare_words])
    df = remove_rare_words(df)
    #df
    ### Removal of Special Characters
    import re

    def remove_spl_characters(text):
        
        text = re.sub('[^a-zA-Z0-9]', ' ', text)
        text = re.sub('\s+', ' ',text)
        
        return text
    df = remove_spl_characters(df)
    #df
    ### Stemming

    #Stemming is the process of reducing a word to its stem that affixes to suffixes and prefixes or to the roots of words known as "lemmas".
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    def stemming(text):
        return " ".join([stemmer.stem(word) for word in text.split()])
            
    df = stemming(df)
    #df
    ### Lemmatization

    #Lemmatization is a text normalization technique used in Natural Language Processing (NLP), that switches any kind of a word to its base root mode. Lemmatization is responsible for grouping different inflected forms of words into the root form, having the same meaning.
    import nltk
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    from nltk import pos_tag
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()

    wordnet_map = {"N": wordnet.NOUN , "V" : wordnet.VERB , "J" : wordnet.ADJ , "A" : wordnet.ADV}

    def lemmatize(text):
        pos_text = pos_tag(text.split())
        return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word,pos in pos_text])
    df = lemmatize(df)
    #df
    import torch
    from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model1 = RobertaModel.from_pretrained("microsoft/codebert-base")
    model1.to(device)

    import pickle
    # create an iterator object with write permission - model.pkl
    with open('model1.pkl', 'wb') as files:
        pickle.dump(model1, files)
    # with open('model1.pkl', 'rb') as f:
    #     data = pickle.load(f)

    # print the Python object
    # print(data)     
        # print('hii')
        # # Read the ipynb file
        # with open('NLP_Automation_Codebert.ipynb',encoding="utf-8") as f:
            
        #     nb = nbformat.read(f, as_version=4)
        #     #print('hii') prob above
        # # Execute the code in the file with the input data
        # ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        # ep.preprocess(nb, {'metadata': {'path': '.'}, 'input_data': input_data})
        # # Get the output of the code
        # #output = nbformat.writes(nb)
        
        # model1_pkl= nbformat.writes(nb)
    
    # return data



# @app.route('/result')
# # ‘/’ URL is bound with hello_world() function.
# def result():
#     return render_template('result.html')


# @app.route('/customnn')
# # ‘/’ URL is bound with hello_world() function.
# def customnn():
#     return render_template('customnn.html')


# @app.route('/', methods=['POST'])
# def my_form_post():
#     text = request.form['input-box']
#     return text


# main driver function
if __name__ == '_main_':

    # run() method of Flask class runs the application
    # on the local development server.
    app.run()
