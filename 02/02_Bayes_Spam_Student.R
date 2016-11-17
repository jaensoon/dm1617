# Lab: Naive Bayes
## Classification of spam SMS messages 

# Libraries --------------------------------------------------------------------
install.packages('tm')
install.packages('wordcloud')
install.packages('gmodels')
library(tm) # build a corpus using the text mining (tm) package
library(wordcloud) # word cloud visualization
library(gmodels) # we'll use CrossTable() from gmodels
library(e1071)  # library for Naive Bayes
library(caret)  # you can also use caret for NB
 
# Importing Data ---------------------------------------------------------------
# read the sms data into the sms data frame
sms_raw <- read.csv("/home/jwendling/Seafile/Study/Master/Semester 2/Data Mining/Lab/dm1617/data/sms_spam.csv", stringsAsFactors = FALSE)

# examine the structure of the sms data
str(sms_raw)

# convert spam/ham to factor.
sms_raw$type <- factor(sms_raw$type)

# examine the type variable more carefully
str(sms_raw$type)
table(sms_raw$type)

# Preprocessing  ---------------------------------------------------------------
sms_corpus <- Corpus(VectorSource(sms_raw$text))

# examine the sms corpus
print(sms_corpus)
inspect(sms_corpus[1:3])

# clean up the corpus using tm_map()
# Preprocessing 
corpus_clean <- tm_map(sms_corpus, tolower)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)
corpus_clean <- tm_map(corpus_clean, PlainTextDocument)

# examine the clean corpus
inspect(sms_corpus[1:3])
inspect(corpus_clean[1:3])

# create a document-term sparse matrix
sms_dtm <- DocumentTermMatrix(corpus_clean)
sms_dtm

# creating training and test datasets
sms_raw_train <- sms_raw[1:4169, ]
sms_raw_test  <- sms_raw[4170:5559, ]

sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test  <- sms_dtm[4170:5559, ]

sms_corpus_train <- corpus_clean[1:4169]
sms_corpus_test  <- corpus_clean[4170:5559]

# check that the proportion of spam is similar
prop.table(table(sms_raw_train$type))
prop.table(table(sms_raw_test$type))

# indicator features for frequent words
# DTMs have more than 7000 columns - that’s way too much. 
# Eliminate words which appear in less than 5 SMS messages (about 0.1%) with
# tm’s findFreqTerms() function. 
# This should reduce the feature-set to a far more manageable number. 
freq_terms <- findFreqTerms(sms_dtm_train, 5)
sms_train <- DocumentTermMatrix(sms_corpus_train, list(dictionary = freq_terms))
sms_test  <- DocumentTermMatrix(sms_corpus_test, list(dictionary = freq_terms))

ncol(sms_train)
ncol(sms_test)

# convert counts to a factor
# Naive Bayes  works on factors, but our DTM only has numerics. 
# We define a function which converts counts to Yes/No factor, 
# and apply it to our reduced matrices.
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}

# apply() convert_counts() to columns of train/test data
# MARGIN = 1 is for rows, and 2 for columns
sms_train <- apply(sms_train, MARGIN = 2, convert_counts)
sms_test  <- apply(sms_test, MARGIN = 2, convert_counts)

# Visualization ----------------------------------------------------------------
wordcloud(sms_corpus_train, min.freq = 30, random.order = FALSE)

# subset the training data into spam and ham groups
spam <- subset(sms_raw_train, type == "spam")
ham  <- subset(sms_raw_train, type == "ham")

wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))



# Training a model on the data -------------------------------------------------
#?train
#sms_train
#ctrl <- trainControl(method="cv", 10)
#sms_classifier1 <- train(sms_raw_train$type~ ., data=sms_raw_train, method = "nb", trControl = ctrl)
sms_classifier1 <- NaiveBayes(sms_train, data = sms_raw_train)
#?NaiveBayes
#sms_classifier1

# Evaluating model performance -------------------------------------------------
sms_test_pred <- predict(sms_classifier1, sms_test)

CrossTable(sms_test_pred, sms_raw_test,
           prop.chisq = FALSE, 
           prop.t = FALSE, 
           prop.r = FALSE, # eliminate cell proprtions
           dnn = c('predicted', 'actual')) # relabels rows+cols

# Improving model performance --------------------------------------------------
# Train a second classifier with Laplace correction
sms_classifier2 <- NaiveBayes(sms_raw_train, data = sms_raw_train, fL=1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)

CrossTable(sms_test_pred2, sms_raw_test,
           prop.chisq = FALSE, 
           prop.t = FALSE, 
           prop.r = FALSE, # eliminate cell proprtions
           dnn = c('predicted', 'actual')) # relabels rows+cols
