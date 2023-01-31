# math156-sentiment-analysis

Authors: [Grant Kinsley](https://github.com/grantKinsley), Jaden Nguyen, Elliot Fouts, Pieter Van Tol

## Introduction
For our final project, our group survied pre-processing methods and neural network architectures for Sentiment Analysis. 
We have included research about two pre-processing methods, Word2Vec and GloVe, and two network
architectures, Convolutional Neural Networks (CNN) and Long Short Term Memory Neural Networks
(LSTM). In addition to our survey of the topic, we used Tensorflow to compare the two models.
We are working with a binary classification dataset from IMDB, which is a labelled dataset 
including positive and negative text reviews of movies. We have included the
simulation results and analysis of our neural network architectures.

## Word Embedding
### Word2Vec
Word vectors are numerical representations of words that preserve the semantic relationship
between them. In fact, word vector algorithms use the context of words to learn their numerical
representations such that words within the same context have similar looking word vectors. There are
numerous usage of word vectors such as language modeling, chatbots, machine translation, question
answering, and it is heavily used in NLP frontiers. The mechanics of Word2vec algorithms first starts
with creating data tuples where each word is represented as one-hot vectors. Then we define a model that
can take those one-hot vectors as inputs & outputs so that they can be trained. A loss function is then
utilized to predict the correct word, which is actually in the context of the input word, to further optimize
the model. Finally, the model is evaluated by making sure similar words have similar word vectors.

### Global Vectors (GloVe)
As we have just discussed in Word2Vec, word vectors put words into a vector space where similar
words cluster together, while different words repel each other. But unlike Word2Vec, GloVe uses global
statistics, in addition to local statistics, in order to obtain word vectors. GloVe is useful because even
though Word2Vec performs well, it is not fundamentally sound since it relies only on local information
and any learning for a certain word is only affected by the surrounding words. Take, for example, “The
dog sat on the log.” Word2Vec would not be able to answer questions such as “is there a correlation
between ‘the’ in context to the words ‘dog’ and ‘log’?” or “is ‘the’ just a stopword?” (Stopwords are
common words that search engines filter out). This is where GloVe improves upon Word2Vec with the use
of global statistics.
GloVe uses the idea of co-occurrence, how often two words appear together globally, to derive
semantic relationships between words. Given a sentence with V words, the co-occurrence matrix (X) for
that sentence is of size V x V, where X_ij denotes how many times word i co-occurs with word j. From
this matrix, we take a third work (k) and can compute the probability of seeing words i and k together
(P_ik) and words j and k together (P_jk). We then use P_ik/P_jk to compute word vectors and define a
mean square cost function to calculate loss.

## Models
### Convolutional Neural Networks
In general, a convolutional neural network (CNN) is a variant of a multi-layer perceptron that
relies on shared parameters, and convolutional layers, and local connections to improve computational
efficiency in high-dimensional machine learning. Since each neuron is connected to a local subset
(receptive field) of neurons in the previous layer, CNNs can account for spatially local relationships
during training.

One common assumption applied with CCNs is that a set of weights of biases (a filter) that is
useful in mapping one region of the input vector will also be applicable to other areas of the input vector.
Under this assumption, CNNs share filters across neurons in a layer rather than computing individual
weights and biases for each. Clearly, this reduces memory usage and computational resources. Also, the
sequential nature of words in a sentence can be understood through spatially local analysis. These
convolutional layers are often followed by a layer which reduces dimensionality by averaging, or taking
the max of subsets of neurons. These layers are referred to as down-sampling layers or pooling layers.

### Long Short Term Memory Neural Networks

LSTM Neural Networks are a form of neural network that help solve several problems other
forms of neural network face. Specifically, they overcome the limitation of traditional neural networks and even Recursive Neural Networks (RNN) not carrying information far enough forward from earlier steps, thus leaving out possibly important and relevant information to later steps.

LSTMs have the same basic structure as Recursive Neural Networks. They pass data through a
sequence of LSTM cells, maintaining information from previous cells and passing it on to later cells.

However, in contrast to RNN, LSTM uses a variety of gates to regulate the passing of information and avoid issues common in RNN as mentioned above. It uses a forget gate to determine what info should be forgotten from past steps, an input gate to determine what new info should be added and an output gate to determine the next hidden state. These gates use sigmoid functions and tanh functions to scale the results in such a way that the gradients eventually applied to the cell state do not vanish or explode as is common in regular neural networks and recursive neural networks. Indeed, if not dealt with, the vanishing or exploding gradient problem will heavily affect an algorithm’s learning ability, an issue LSTMs do not face.

### Model Architectures
While constructing the architecture of the neural networks, we had different options for each of the layers of the neural networks. Our models use the Sequential class, which is a linear stack of layers. We first add the default Embedding layer provided by keras, which turns positive integers into dense vectors of fixed size. From here, the structures of our CNN and LSTM models diverge. In the CNN model, we add a convolution layer, which produces a tensor of outputs. Next, we add a pooling layer, global max pooling and global average pooling, to down sample the feature maps, so that we can eventually have a fully connected layer. We decided to use global max pooling since it gave us better results, which we think is because max pooling preserves the stark features between different data points. Next, for our last hidden layer, we use ReLU as the activation function, to avoid the vanishing
gradient problem that CNN usually runs into. The vanishing gradient problem occurs when the gradient of
the error function disappears, so the model cannot make progress with its learning. ReLU solves this issue
since the gradient will never fully vanish, since it is always 0 or 1. Finally for the output layer, we use the
sigmoid function as the final activation function, since we are using this training model on binary
classification.
In the LSTM model, we add an LSTM layer, which is a type of recurrent neural network layer
that avoids long-term dependency problems that other recurrent neural networks face. We then tested
LSTM with a global max pooling layer and also without a pooling layer. After trying both options, we
observed that the testing accuracy increased with the use of the global max pooling layer. We then use
ReLU as the activation function for the next hidden layer. After that, we add a dropout layer that
randomly sets inputs to 0 to prevent overfitting. Finally, for the output layer, we use the sigmoid function
as the final activation function, for the same reason we used it in the output layer for CNN.

## Simulation Results
See GitHub repository for results:https://github.com/grantKinsley/Sentiment_Analysis

## Analysis of Simulation Results 
From the simulation results, we observed that our models exhibit slight overfitting, due to the
high training accuracy, but lower testing accuracy. Between CNN and LSTM, when the batch sizes are the
same, both models perform about the same. Finally, for the LSTM model, we observed that the model
performs better when we add the global max pooling layer to the model rather than when there was no
pooling.

## Challenges and Difficulties
As a group, we collectively found it difficult to fully understand the concepts and theory behind each of the topics we researched. For example, we only went over convolutional neural networks (CNNs) in class but not recursive neural networks (RNNs), so the math behind the LSTM model was hard to
follow. Another example is the global vector (GloVe) word embeddings. One of the difficulties with GloVe is that there is a dimensional mismatch between the word vectors (high-dimensional) and the probabilities computed to represent the vectors (scalar). Another difficulty with GloVe is that there are
three variables to account for when computing the loss function, so there needs to be a method that reduces it to just two variables. During model evaluation, we ran into issues with overfitting in our LSTM model. However, adding a dropout layer significantly boosted our training accuracy.

References
Research links
https://en.wikipedia.org/wiki/Convolutional_neural_network#Convolutional_layers
https://medium.com/saarthi-ai/sentence-classification-using-convolutional-neural-networks-ddad72c
c
https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-understanding-word2v
ec-e0128a460f0f
https://towardsdatascience.com/light-on-math-ml-intuitive-guide-to-understanding-glove-embeddings-b
b4f19c
https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb
bf

Dataset
https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data

CNN Code
https://realpython.com/python-keras-text-classification/#convolutional-neural-networks-cnn
https://towardsdatascience.com/cnn-sentiment-analysis-9b1771e7cdd

LSTM Code
https://cnvrg.io/pytorch-lstm/
https://towardsdatascience.com/understanding-lstm-and-its-quick-implementation-in-keras-for-sentiment-
analysis-af410fd85b
https://www.kaggle.com/nilanml/imdb-review-deep-model-94-89-accuracy/notebook
