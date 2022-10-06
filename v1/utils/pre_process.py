import config
import re
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


class Preprocess:

    def __init__(self) -> None:
        pass


    def clean_text(self,text:str)->str:
        """
        This method will be used to clean text
        Below is the text cleaning that will be done

        Attributes:
            text (str) : input string
        Returns:
            inp_str (str) : cleaned text
        
        """
        inp_str = re.sub('\n','', text)
    
        inp_str = re.sub('http:\S+','', inp_str)#removing urls

        # removing chars such as slashes ,hash ,.,? etc
        inp_str=re.sub("\'|\"|\,|\.|\?|\+|\-|\/|\=|\(|\)|\n|\:|\<|\>|\@|\!|\/|\#"," ",inp_str) 
        inp_str=inp_str.replace('\\',' ')
        
        
        inp_str=re.sub(r'\b\w{1,3}\b',' ',inp_str)# removing words having length less than equal to 2
        
        
        inp_str=re.sub(r' +'," ",inp_str)# removing extra spaces from string
    
        return inp_str
    
    def preprocess(self,doc:str,stem=True)->str:

    
        """
        1.change  all the word to lower case
        2. tokenize
        3.remove stop word
        4. reduce to root word

        Argument:
            doc (str) : input text
        Returns:
            document (str) : output string

         """
        
        stemmer = PorterStemmer() #initializing stemmer 
        wordnet_lemmatizer = WordNetLemmatizer() #initializing lemmatizer

        doc=doc.lower()# 1st step
        
        words=word_tokenize(doc)
        
        words=[word for word in words if word not in stopwords.words('english') ]
        
        if stem:
            words= [stemmer.stem(word) for word in words]
        
        else:
            words=[wordnet_lemmatizer.lemmatize(word, pos='v') for word in words]
            
        
            # join words to make sentence
        document = " ".join(words)

        return document

