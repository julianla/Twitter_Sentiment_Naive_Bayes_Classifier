# Twitter Sentiment Naive Bayes Classifier
A Multinomial Naive Bayes Classifier trained on the training data of the Sentiment140 project (sentiment140.com). 
The training data consists of 1,600,000 tweets that are classified as either negative or positive.


## Usage

First, download the training data, which is available here: [Sentiment140](http://help.sentiment140.com/for-students). Unzip the folder and move the file `training.1600000.processed.noemoticon.csv` to your project folder. Then, run the script in your command line:

`python NB.py Number_of_training_examples_you_want_to_use`

In addition to saving the classifier in a `sav` file, the code also prints
the precision score, classification report and the confusion matrix based on the test data.
The test data consists of the remaining examples in the training data that were not
chosen to train the classifier.


Tested with Python 2.7.

