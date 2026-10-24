from string import punctuation, digits
import numpy as np
import random
def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices

def hinge_loss_single(feature_vector, label, theta, theta_0):
    z = label*(np.dot(feature_vector,theta) +theta_0)
    hinge_loss = max(0, 1-z)
    return hinge_loss
def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    rows = feature_matrix.shape[0]
    hinge_loss = 0
    for i in range(rows):
        hinge_loss += hinge_loss_single(feature_matrix[i], labels[i], theta, theta_0)
    return hinge_loss/rows


def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    z = label*(np.dot(feature_vector,current_theta) +current_theta_0)
    if z<=0: #the solution has used a for loop
        current_theta = np.add(current_theta, label*feature_vector)
        current_theta_0 = current_theta_0 + label
        z = label*(np.dot(feature_vector,current_theta) +current_theta_0)
    return (current_theta, current_theta_0)

def perceptron(feature_matrix, labels, T):
    nsamples, nfeatures = feature_matrix.shape 
    current_theta = np.zeros(nfeatures)
    current_theta_0 = 0
    for t in range(T):
        for i in get_order(nsamples):
            current_theta, current_theta_0 = perceptron_single_step_update(feature_matrix[i],labels[i],current_theta,current_theta_0)

    return (current_theta, current_theta_0)

def average_perceptron(feature_matrix, labels, T):
    nsamples, nfeatures = feature_matrix.shape 
    current_theta = np.zeros(nfeatures)
    current_theta_0 = 0
    theta = np.zeros(nfeatures)
    theta_0 = 0
    for t in range(T):
        for i in get_order(nsamples):
            current_theta, current_theta_0 = perceptron_single_step_update(feature_matrix[i],labels[i],current_theta,current_theta_0)
            theta += current_theta
            theta_0+=current_theta_0
    return (theta/(nsamples*T), theta_0/(nsamples*T))
def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        theta,
        theta_0):
    z = label*(np.dot(theta, feature_vector) +theta_0)
    if z<=1:
        theta = np.add((1-L*eta)*theta, eta*label*feature_vector)
        theta_0 = theta_0 + label*eta
    else:
        theta = (1-L*eta)*theta
    return (theta, theta_0)

def pegasos(feature_matrix, labels, T, L):
    """
    nsamples, nfeatures = feature_matrix.shape
    t = 0
    theta = np.zeros(nfeatures)
    theta_0 = 0
    for i in range(T):
        
        for j in get_order(nsamples):
            t= t + 1
            eta = 1/(np.sqrt(t))
            theta, theta_0 = pegasos_single_step_update(feature_matrix[j],labels[j],L,eta,theta,theta_0)
    return (theta, theta_0)
def classify(feature_matrix, theta, theta_0):
    nsamples, nfeatures = feature_matrix.shape
    class_fnc = np.zeros(nsamples)
    for i in range(nsamples):
        if (np.dot(feature_matrix[i], theta) + theta_0)>0:
            class_fnc[i] =  1
        else:
            class_fnc[i] = - 1
    return class_fnc
def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    t_theta, t_theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    train_classify = classify(train_feature_matrix, t_theta, t_theta_0)
    train_accuracy = accuracy(train_classify, train_labels)
    
    val_theta, val_theta_0 = classifier(val_feature_matrix, val_labels, **kwargs)
    val_classify = classify(val_feature_matrix, t_theta, t_theta_0)
    val_accuracy = accuracy(val_classify, val_labels)
    return (train_accuracy, val_accuracy)
def extract_words(text):
    for c in punctuation + digits:
        text = text.replace(c, ' ' + c + ' ')
    return text.lower().split()



def bag_of_words(texts, stopwords_data, remove_stopword=True):
    indices_by_word = {} 
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word in indices_by_word:continue
            if word in stopwords_data: 
                continue
            indices_by_word[word] = len(indices_by_word)
    return indices_by_word

def extract_bow_feature_vectors(reviews, indices_by_word, binarize=True):
    feature_matrix = np.zeros([len(reviews), len(indices_by_word)], dtype=np.float64)
    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word not in indices_by_word: continue
            feature_matrix[i, indices_by_word[word]] += 1
    return feature_matrix

def accuracy(preds, targets):
    return (preds == targets).mean()
