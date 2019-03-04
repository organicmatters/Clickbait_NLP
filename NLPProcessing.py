import time
import os
import requests
import re
import nltk
import string
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF

chromedriver = "/Applications/chromedriver" # path to the chromedriver executable
os.environ["webdriver.chrome.driver"] = chromedriver

class GetArticles:
    """This class provides a pipeline to scrape articles from a website

    Args:
        driver (:obj): This is an instance of a selenium driver
        url (str): The url of the site to be scraped

    """
    
    def __init__(self):
        
        self.driver = None
        self.url = None
        
    def create_driver(self, url: str):
        """Creates a selenium driver and gets the url

        Args:
            url (str): site to be scraped

        Returns:
            Nothing

        """
        
        self.url = url
        driver = webdriver.Chrome(chromedriver)
        self.driver = driver
        return self.driver.get(url)
    
    def scroll(self):
        """Scrolls to the bottom of a long web page for a max of 30 seconds"""

        scroll_pause_time = 4

        # Get scroll height
        last_height = self.driver.execute_script("return document.body.scrollHeight")

        t_end = time.time() + 30

        while time.time() < t_end:
            # Scroll down to bottom
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait to load page
            time.sleep(scroll_pause_time)

            # Calculate new scroll height and compare with last scroll height
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
    def get_site_links(self):
        """ Creates an article list from the home page of a website

        Returns:
            A list of article urls

        """
        soup = BeautifulSoup(self.driver.page_source, "lxml")
        article_list = []
        for link in soup.find_all('a'): 
            try:
                if self.url in link['href']:
                    article_list.append(link['href'])
            except:
                continue
        article_list = list(set(article_list))
        
        return article_list
        
        
    def get_articles(self, article_list: list):
        """returns the article text for each article in a list of articles

                Args:
                    article_list(list): list of urls to get article text from

                Returns:
                    A dictionary of of the articles

        """
        article_dict = {}

        #The slicing was added below to test code
        for article in article_list[0:10]:

            response = requests.get(article)
            page = response.text
            soup = BeautifulSoup(page, "lxml")
            article_title = soup.find('h1')
            article_text = soup.find_all(['h2', 'p'])

            if len(article_text) > 3:

                article_dict[article_title] = article_text

        return article_dict
    
    def create_article_dict(self, url: str):
        """Creates a dictionary of all of the articles from a web page

                Args:
                    url(str): the url for the web page

                Returns:
                    A dictionary of of the articles

        """
    
        self.create_driver(url)
        self.scroll()
        article_list = self.get_site_links()
        
        return self.get_articles(article_list)

    
class NLPPipeline:
    """This class provides a pipeline for NLP models

    Args:
        nltk_stopwords (list): A list of all of the stop words
        topic_model (:obj): Sklearn topic model
        model (:obj): fit topic model to a vectorized corpus
        vectorizer (sklearn.feature_extraction.text.CountVectorizer): Sklearn vectorizer object
        vectorized_corpus (scipy.sparse.csr.csr_matrix): The vectorized corpus


    """
    
    def __init__(self, vectorizer=None, topic_model=None):

        self.nltk_stop_words = set(stopwords.words('english'))
        if not vectorizer:
            vectorizer = CountVectorizer(stop_words=self.nltk_stop_words, min_df=15, max_df=0.25, ngram_range=(1, 3))
        self.topic_model = None
        self.model = None
        self.vectorizer = vectorizer
        self.vectorized_corpus = None
        
    def remove_html(self, article: str):
        """Removes html from an article

                Args:
                    article(str): the raw html from a web page

                Returns:
                    str: Article text

        """

        text = article['text']
        soup = BeautifulSoup(text, "lxml")
        article_text = soup.get_text()[1:-1]

        return article_text
    
    def text_cleaning(self, article: str):
        """Cleans a string of text for tokenization

                Args:
                    article(str): the raw text from a web page

                Returns:
                    str: Article text

        """
    
        clean_text = re.sub('[%s]' % re.escape(string.punctuation), ' ', article)
        clean_text = re.sub('\w*\d\w*', ' ', clean_text)
        clean_text = clean_text.lower() 

        return clean_text

    def text_lemmatizing(self, article):
        """Lemmatizes a string of text

                Args:
                    article(str): Raw text

                Returns:
                    str: raw text

        """
    
        lemmatized_word_list = []
        words = word_tokenize(article)
        wordnet_lemmatizer = WordNetLemmatizer()
        for word in words:
            lemmatized_word = wordnet_lemmatizer.lemmatize(word, pos='v')
            lemmatized_word_list.append(lemmatized_word)

        lemmatized_word_string = ' '.join(lemmatized_word_list)
        
        return lemmatized_word_string
        
    def vectorize(self, articles: str):
        """Vectorizes a string of text

                 Args:
                     articles(str): Raw text

         """
        
        cleaned_article_corpus = []
        for article in articles:
    
            article_no_html = self.remove_html(article)
            clean_article = self.text_cleaning(article_no_html)
            lemmatized_article = self.text_lemmatizing(clean_article)
            cleaned_article_corpus.append(lemmatized_article)
        
        self.vectorized_corpus = self.vectorizer.fit_transform(cleaned_article_corpus)
        
    def fit(self, topic_model=None):
        """Performs topic modeling on a vectorized text corpus

                 Args:
                     topic_model(:obj): Sklearn topic model

         """
        
        if not topic_model:
            topic_model = self.topic_model
        else:
            self.topic_model = topic_model
        
        self.model = self.topic_model.fit_transform(self.vectorized_corpus)
        
    def display_topics(self, no_top_words: int, topic_names=None):
        """Displays the topics for a model

                 Args:
                     no_top_words (int): The number of top words in each category to display
                     topic_names (list): A list of the topic names to display

                 Returns:
                     Prints the topics

         """
        for ix, topic in enumerate(self.topic_model.components_):
            if not topic_names or not topic_names[ix]:
                print("\nTopic ", ix)
            else:
                print("\nTopic: '",topic_names[ix],"'")
            print(", ".join([self.vectorizer.get_feature_names()[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]]))
        