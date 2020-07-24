# Get Stanford Sentiment Treebank
if hash wget 2>/dev/null; then
  wget http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
else
  curl -L http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip -o stanfordSentimentTreebank.zip
fi
unzip stanfordSentimentTreebank.zip
rm stanfordSentimentTreebank.zip
