import numpy as np
import pickle, random, re, sys
from sklearn.metrics import average_precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

def tweets_processing(filecontent, first, last):
  # Prepares tweets for classifier

  features = []
  labels = []
  for tweet in filecontent[first:last]:
    tweet = unicode(tweet, errors='ignore') # Convert tweets to unicode
    tweet  = re.sub(',"[0|1|2|3|4|5|6|7|8|9].+","', '', tweet) # Delete everything except for the tweet and the sentiment
    tweet = re.sub('((www\..+)|(https?://.+))','URL', tweet) # Normalize URLs
    tweet = re.sub('@[^\s]+','at_user',tweet) # Normalize users
    tweet = re.sub('[\s]+', ' ', tweet) # Delete multiple white spaces
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # DeletesHashtags
    tweet = tweet.strip() # Trim leading and trailing white spaces
    tweet  = re.sub('\n|"|@', '', tweet) # Delete \n, " and @
    tweet = tweet.lower() # Convert tweets to lowercase
    lab,feat =  tweet[0], tweet[1:] # Separate labels and features
    features.append(feat)
    labels.append(lab)
  labels = [int(float(l)) for l in labels]
  labels = [1 if l==4 else l for l in labels] # Just for convenience: Change the positive value from 4 to 0
  return features, labels
  
def classifier(features, labels, test_features, test_labels):
  """Implement the Multiomial Naive Bayes Classifier and prints
  the precision score, classification report
  and confusion matrix based on the specified test data. 
  Then use pickle to save the predicted values to MultinomialNB.pkl."""

  text_clf = Pipeline([('hv', HashingVectorizer(non_negative=True, stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
  ])
  text_clf.fit(features, labels)
  pred_lab = text_clf.predict(test_features)
  print average_precision_score(test_labels,pred_lab)
  print metrics.classification_report(test_labels, pred_lab)
  print metrics.confusion_matrix(test_labels, pred_lab)
  with open('MultinomialNB.pkl', 'wb') as fout:
    pickle.dump(text_clf, fout)

def save_classifier(classifier):
  # Save the classifier to MultinomialNB.sav
  
 filename = 'MultinomialNB.sav'
 pickle.dump(classifier, open(filename, 'wb'))
   
if __name__ == "__main__":   
  file = open('training.1600000.processed.noemoticon.csv', 'rU')
  filecontent = file.readlines()
  random.shuffle(filecontent) # Shuffle the tweets randomly
  if int(sys.argv[1]) + 1 >= 1600000:
    sys.exit('Error: Test data must have at least 1 example.')
  features, labels = tweets_processing(filecontent,0,int(sys.argv[1]))
  test_features, test_labels = tweets_processing(filecontent,int(sys.argv[1])+1,1600000)
  classifier = classifier(features, labels, test_features, test_labels)    
  save_classifier(classifier)
  
















