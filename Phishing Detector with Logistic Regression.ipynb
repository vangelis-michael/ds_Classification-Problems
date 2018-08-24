{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# PROJECT 2: Phishing Detector with Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#CellStrat - Load the phishing website dataset (this dataset has 11000+ data samples). Each sample has 30 website\n",
    "#parameters and then a class label identifying it as a phishing website or not (1 or -1).\n",
    "\n",
    "#The task is to train a KNN and Logistic Regression classifier which can detect phishing websites.\n",
    "\n",
    "#Using 70% data for training and 30% for testing.\n",
    "\n",
    "#Train the model using training data and then use it to predict the test data.\n",
    "\n",
    "#Then print the count of misclassified samples in the test data prediction as well as the accuracy score of this prediction."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Import modules"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn import datasets\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Define Input and Output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "phishing = np.loadtxt('phishing.txt', delimiter=',')\n",
    "\n",
    "#attribute having_IP_Address  { -1,1 }\n",
    "#attribute URL_Length   { 1,0,-1 }\n",
    "#attribute Shortining_Service { 1,-1 }\n",
    "#attribute having_At_Symbol   { 1,-1 }\n",
    "#attribute double_slash_redirecting { -1,1 }\n",
    "#attribute Prefix_Suffix  { -1,1 }\n",
    "#attribute having_Sub_Domain  { -1,0,1 }\n",
    "#attribute SSLfinal_State  { -1,1,0 }\n",
    "#attribute Domain_registeration_length { -1,1 }\n",
    "#attribute Favicon { 1,-1 }\n",
    "#attribute port { 1,-1 }\n",
    "#attribute HTTPS_token { -1,1 }\n",
    "#attribute Request_URL  { 1,-1 }\n",
    "#attribute URL_of_Anchor { -1,0,1 }\n",
    "#attribute Links_in_tags { 1,-1,0 }\n",
    "#attribute SFH  { -1,1,0 }\n",
    "#attribute Submitting_to_email { -1,1 }\n",
    "#attribute Abnormal_URL { -1,1 }\n",
    "#attribute Redirect  { 0,1 }\n",
    "#attribute on_mouseover  { 1,-1 }\n",
    "#attribute RightClick  { 1,-1 }\n",
    "#attribute popUpWidnow  { 1,-1 }\n",
    "#attribute Iframe { 1,-1 }\n",
    "#attribute age_of_domain  { -1,1 }\n",
    "#attribute DNSRecord   { -1,1 }\n",
    "#attribute web_traffic  { -1,0,1 }\n",
    "#attribute Page_Rank { -1,1 }\n",
    "#attribute Google_Index { 1,-1 }\n",
    "#attribute Links_pointing_to_page { 1,0,-1 }\n",
    "#attribute Statistical_report { -1,1 }\n",
    "#attribute Result  { -1,1 }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(11055L, 31L)"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "phishing.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Create X and Y data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "('Class labels:', array([-1.,  1.]))\n"
     ]
    }
   ],
   "source": [
    "#X = phishing[:, [1, 5]]\n",
    "X = phishing[:, 0:30]\n",
    "y = phishing[:,30:31]\n",
    "print('Class labels:', np.unique(y))\n",
    "\n",
    "\n",
    "# Splitting data into 70% training and 30% test data:\n",
    "X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,\n",
    "                                                    random_state=1, stratify=y)\n",
    "y_train = np.array(y_train.ravel())\n",
    "y_test = np.array(y_test.ravel())\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Train and Evaluate Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Misclassified samples: 211\n",
      "Accuracy: 0.94\n",
      "Accuracy: 0.94\n"
     ]
    }
   ],
   "source": [
    "#from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')\n",
    "knn.fit(X_train, y_train)\n",
    "y_pred = knn.predict(X_test)\n",
    "\n",
    "print('Misclassified samples: %d' % (y_test != y_pred).sum())\n",
    "print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))\n",
    "print('Accuracy: %.2f' % knn.score(X_test, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Exercise 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Build a phishing website classifier using Logistic Regression with “C” parameter = 100.\n",
    "#Use 70% of data as training data and remaining 30% as test data.\n",
    "#Print count of misclassified samples in the test data prediction as\n",
    "#well as the accuracy score of the model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Logistic Regression import and build\n",
    "\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "lr = LogisticRegression(C = 100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "('Class labels:', array([-1.,  1.]))\n"
     ]
    }
   ],
   "source": [
    "#Already done above\n",
    "#Splitting the dataset:\n",
    "\n",
    "X = phishing[:, 0:30]\n",
    "y = phishing[:,30:31]\n",
    "print('Class labels:', np.unique(y))\n",
    "\n",
    "# Splitting data into 70% training and 30% test data:\n",
    "X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,\n",
    "                                                    random_state=1, stratify=y)\n",
    "y_train = np.array(y_train.ravel())\n",
    "y_test = np.array(y_test.ravel())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Misclassified samples: 249\n",
      "Accuracy of the test with predicted: 0.92\n",
      "Accuracy Score of the model on the test set: 0.92\n"
     ]
    }
   ],
   "source": [
    "# Logistic Regression fit and prediction\n",
    "lr.fit(X_train, y_train)\n",
    "y_pred1 = lr.predict(X_test)\n",
    "\n",
    "# Misclassification and accuracy score\n",
    "print('Misclassified samples: %d' % (y_test != y_pred1).sum())\n",
    "print('Accuracy of the test with predicted: %.2f' % accuracy_score(y_test, y_pred1))\n",
    "print('Accuracy Score of the model on the test set: %.2f' % lr.score(X_test, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# INFERENCE: The Logistic Regression model with 'C' parameter = 100 doesn't perform as good as KNN on the test set and predictions\n",
    "# We can find out how this matches with the train set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy Score of the model on the train set(LogitR): 0.93\n",
      "Accuracy Score of the model on the train set(KNN): 0.96\n"
     ]
    }
   ],
   "source": [
    "print('Accuracy Score of the model on the train set(LogitR): %.2f' % lr.score(X_train, y_train))\n",
    "print('Accuracy Score of the model on the train set(KNN): %.2f' % knn.score(X_train, y_train))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Exercise 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Train with only two input parameters - parameter Prefix_Suffix and 13 URL_of_Anchor.\n",
    "#Check accuracy using the test data and compare the accuracy with the previous value.\n",
    "#Plot the test samples along with the decision boundary when trained with index 5 and index 13 parameters."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Selecting the columns Prefix_Suffix and 13 URL_of_Anchor\n",
    "# Prefix_Suffix = Column 5\n",
    "# URL_of_Anchor = Column 13\n",
    "X = phishing[:, [5, 13]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(11055L, 2L)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Splitting the new X and y into 70% training and 30% test data:\n",
    "X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,\n",
    "                                                    random_state=1, stratify=y)\n",
    "y_train = np.array(y_train.ravel())\n",
    "y_test = np.array(y_test.ravel())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Misclassified samples: 543\n",
      "Accuracy of the test with predicted: 0.84\n",
      "Accuracy Score of the model on the test set: 0.84\n"
     ]
    }
   ],
   "source": [
    "# Using Logistic Regression still using the parameter \"C\" = 100\n",
    "lr.fit(X_train, y_train)\n",
    "y_pred2 = lr.predict(X_test)\n",
    "\n",
    "print('Misclassified samples: %d' % (y_test != y_pred2).sum())\n",
    "print('Accuracy of the test with predicted: %.2f' % accuracy_score(y_test, y_pred2))\n",
    "print('Accuracy Score of the model on the test set: %.2f' % lr.score(X_test, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Training with just the Prefix_Suffix and URL_of_Anchor shows a reduced accuracy though inversely shows that these 2 features\n",
    "# have high correlation with whether it is a phishing account or not a phishing account.\n",
    "# We can also see that Logistic Regression with just the 2 features misclassified 543 samples showing it definitely still needs\n",
    "# the rest of the data to properly identify the target."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Import some libraries for plotting, the ListedColormap for color maps and the %matplotlib inline for inline plot printing\n",
    "import matplotlib\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from matplotlib.colors import ListedColormap\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Scale the data to allow for stacking and avoid improper weightage on the LR formula\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "sc = StandardScaler()\n",
    "sc.fit(X_train)\n",
    "X_train_std = sc.transform(X_train)\n",
    "X_test_std = sc.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create a decision regions function for plotting the decision regions \n",
    "def plot_decision_regions(X, y, classifier, test_idx=None, resolution = 0.02):\n",
    "    #Setup marker generation and color map\n",
    "    markers = (\"s\", \"x\", \"o,\", \"^\", \"v\")\n",
    "    colors = (\"red\", \"blue\", \"lightgreen\", \"gray\", \"cyan\")\n",
    "    cmap = ListedColormap(colors[:len(np.unique(y))])\n",
    "    \n",
    "    #Plot the decision surface\n",
    "    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1\n",
    "    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1\n",
    "    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),\n",
    "                          np.arange(x2_min, x2_max, resolution))\n",
    "    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)\n",
    "    Z = Z.reshape(xx1.shape)\n",
    "    plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap=cmap)\n",
    "    plt.xlim(xx1.min(), xx1.max())\n",
    "    plt.ylim(xx2.min(), xx2.max())\n",
    "    \n",
    "    for idx, cl in enumerate(np.unique(y)):\n",
    "        plt.scatter(x=X[y == cl, 0],\n",
    "                   y=X[y == cl, 1],\n",
    "                   alpha = 0.8,\n",
    "                   c = cmap(idx),\n",
    "                   marker=markers[idx],\n",
    "                   label=cl,\n",
    "                   edgecolor='black')\n",
    "        #Highlight test samples\n",
    "    if test_idx:\n",
    "        #plot all samples\n",
    "        X_test, y_test = X[test_idx, :], y[test_idx]\n",
    "        \n",
    "        plt.scatter(X_test[:, 0],\n",
    "                   X_test[:, 1],\n",
    "                   c = '',\n",
    "                   edgecolor='black',\n",
    "                   alpha = 1.0,\n",
    "                   linewidth=1,\n",
    "                   marker = 'o',\n",
    "                   s=100,\n",
    "                   label='test set')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Stack up the sets \n",
    "X_combined_std = np.vstack((X_train_std, X_test_std))\n",
    "y_combined = np.hstack((y_train, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression(C=100.0, class_weight=None, dual=False, fit_intercept=True,\n",
       "          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,\n",
       "          penalty='l2', random_state=1, solver='liblinear', tol=0.0001,\n",
       "          verbose=0, warm_start=False)"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Training the logistic regression model with scikit_learn\n",
    "lr = LogisticRegression(C=100.0, random_state = 1)\n",
    "lr.fit(X_train_std, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA3oAAAGDCAYAAAB9UWKAAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAAPYQAAD2EBqD+naQAAIABJREFUeJzs3X+cVGXd//H3RxBj5YeGgG6QomlmlqQ2mTfalllRYZZ1\nS3fbHVSWgkj1rcEf+Ssqbva+Tc1flaXYTSkV6Y0ZmpmLa5brryUNi0JQCGPRlB8uiML1/eO6hj17\ndmZnZndmz8zs6/l4zGN3zlxzzmfOnHPN+ZzrXNcx55wAAAAAALVjj6QDAAAAAACUFokeAAAAANQY\nEj0AAAAAqDEkegAAAABQY0j0AAAAAKDGkOgBAAAAQI0h0QMAAACAGkOiBwAAAAA1hkQPAAAAAGpM\nWRI9M2swM2dmDeWYf2+Z2afN7C9m9oqZvdiL9x8UPtdXyxEfSi/ynU1LOpZ8zGyBma0pouzWMsQw\nLayvY0s97yKXf1AlzT+855JyxJRnuWX5nvtLUuut1Mys2cyeSDqO/lJMXVSGZZe7DlhjZguKKPur\nPiyr5Nt/rexT5VLothv26eYSL7vk8wT6qqhEL1IBZx7bzWylmV1tZmNLEZCZfbAclZiZHS5pgaRV\nks6Q9IX+jqFQZlZnZpeUO1FO+nOiZ+XcDsxsRjUkvwC6M7P6UDdMTDqWamdmR4R1eVDSsVSykPS6\nLI/vJR3bQGdmKTO71sweCQ0ZLke5oWb2IzN7wsw2mdlWM1tuZrPNbM/+jrvaVGu9O7iX77tI0mpJ\nr5E0SdJZkj5oZkc65zr6GNMHJc2UdEkf5xPXIJ/YznbO/T2hGApVJ+ni8H9zGZeT9OdEV2eo68mX\ncm4HMyQ9J3/yYyD5X0m3SHq5yPcNlfRq6cMBeqVevm5YI6mtDPOP10W15I2SdkWeHyG/Lpvl12cp\n1Vq90Sbpsti0lUkEUgLvSzqAEvqgpM9L+pOkpyQdlqPcUElvlvRr+W19l6TjJV0u6R2S/qPcgVa5\ncte7ZdHbRG+pc+7h8P8Pzex5SV+R9BFJN5ckstIbE/4Wfckm0B+cc68kHUOtc87tlLSzF+/bXoZw\nUAQz29s591LScVQjM6sr5iRsLddFzrliT/L0ZVm1Vm/8wzm3MOkgSsE5tyPpGEroOknznXPbzOxq\n5Uj0nHP/knRcbPL3zGyTpLPN7CvOuX+WOda8iq2vql25f9tKdcbud+HvhJ4KmdknQtPyNjN7zswW\nmtnrIq8vkG9hylyH7nI1QcfmO8PM/mxmL5vZejO7xsz2iby+RtKl4enGnq5xLzQGM/uCma0Ky3zI\nzN6epczhZvYLM/tXuMz1YTM7Jc9nOUjSxvD04kgMl0TK5J2vme1pZheb2d9CmefN7H4zO7mYzxmb\n57Fmdlf47raZ2WozuyFW5qtm9kBY3rbwfX88y7xcuOT3E2a2IpT9g5m9Jbz+RTP7e4i9OX5ZTZj2\nhJkdE5aXiefMnj5DKddhjvnuY2Y7zeycyLT9zGxXeL9Fpl9nZv+MPN/dt6CQ7SCUe52Z3Wb+EoyN\nZvY/ZjYoz2dfI39W712R+TbHiu1lZt8J83zJzG41s9FZ5jXZzFpCmS1mdoeZvbmn5Ufe+2Yz+134\n7taZ2deVo04qdDnhe/1ZiHubmf3VzL4Veb1b/58Ct+ts6/5tZrbUzDaH9X+PmR0XK5NZ3r8Vsj57\nWFcHhxhfMl/HXRTdlkKZvc3sMjNba75e+qv5/TG6zeXssxr/jOYvUXFm9oawbb5o/nKfG82sLvbe\nvczs8vD5tpjZEjMbl2UZB5q/xOivYV0/b2Y/t+77d2a9vSuUb5e0zszeHaZ/NMu8/yO89s4c67DX\n+2Zk+hFmdq+ZdZjZP8wsnaXMXmZ2qfn66+XwfTSZ2V5Z1vfVZnaq+brsZfO/Yx/IFn/kfQ2SHgpP\nb7TOfXhaeD1aN95nZh2Svh1e+0jYd9aH5a0yswstVmdYrJ9TZLv5qpXwt8+KqANi7zslxPPWyLTT\nwrRfxso+aWaLIs9399EL6+zn4aV7I+uyITaPSWbWGj7LU2b2n/liDO8r+z4Vyr7OzG4wsw2R7eiz\nsTI3hfjfFJt+l5m9YGb1BX6mIWa2dyFlY+/5hvnjgU3m67EWM3t3rFyx21lm39ke/narF3qIqUt/\nOuscW+LfzeyCsD1uN1+vvyHL+zPxbQvbxgk5lpO3PjCz6WHZ8e/s/DD9gz19FufcBufctkI/exZr\nwt99eipknfXyiWb2ffP15mYz+7GZ7RsrW2hdU4r6KjOPt5rZMvP1898tHHua/x150DqPCd6b5bP1\nuA9Znno3lHmHmd0ZtvGOEMu/xZaTqQOOMLOfmtkLku4Pr+1vvi5YF2J41sz+z/p6WblzruCHpGmS\nnKRjY9PPCdO/GJ43hOcNWd7bKulLkuZJ6pC/BHSfUOadkn4TyjVmHnliuiSUv1vS2ZKukr9UolXS\n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\n5DZS0r96KmBme5nZiMxD0vD+CQ1A0RobtehNl+7us0fLHgCglpHoAVmY2RskzZL0/TxFz5O0KfJY\nV+bQAPRFOq2W2YvVoGVqvpI+ewCA2kWih5pmZv9lZi7P4/DYe14n6U5JP3fOXZ9nEfPkW/4yj3Fl\n+SAASieV0txrR/k+e7TuAQBqlDnnko4BKBszGy1pVJ5iTznndoTy9ZKaJf1R0jTn3K4ilzdC0qZN\nt9yiEXV1vYgYQL9qbdWFCw5R86aJ0lETlU4nHRAwcHV0bNbUqSMlaaRzbnPS8QDVjkQPCEJL3r2S\nHpHU6Jzb2Yt5kOgB1aipSScsv0oaN17p+fnODQEoBxI9oLS4dBPQ7iSvWdIzkr4qabSZ7W9m+yca\nGID+kU77SzkBAKgRJHqAd7KkN0g6SX5AlWcjj//f3t0H2VXXdxx/fyCAFuIjTypDUWvFp8qoRbQd\nTIc6DhVF+KNiYSYJg0+AilahFWtRpgi2MI6AOFU0WHTKMHbwadBqhda2AtYhYokPg0QSjCEgAyQN\nEIRv/zhn8bqzSTa7y/6y575fM3d27+/e87ufPcnM5pvv7/yOpHFx+1o+unyV1+xJkhY8Cz0JqKoV\nVZWpHq2zSZonp5/Odz6/liWPv6HbkfOMX7VOJEnSjFnoSZI04tEdOe3uSZIWMAs9SZImG+3uXbKK\nyy9vHUiSpB1joSdJ0lac/YmnsuTxtvQkSQuPhZ4kSdtw6OIfse5qr9mTJC0sFnqSJG3D6877Y75z\n5Ed4+oYb+ehyl3FKkhYGCz1JkrbnhBO44rP3c/q+l9ndkyQtCBZ6kiRNk909SdJCYaEnSdKOsLsn\nSVoALPQkSZqBR7t7/MJ77UmSdjoWepIkzcaGDVz7Rbt6kqSdy6LWAaQhum/z5tYRJM2HY4/lUw99\ngmXXLuWc976Y0z78lNaJpAVr8+b7WkeQBiVV1TqDNBhJngHc3jqHJEkL2AFV9YvWIaSFzkJPmkNJ\nAjwd2Ng6yySL6QrQA9j5sg2d574Nz3s7nvt2hnDuFwPryn+gSrPm0k1pDvW/mHa6/4Xs6k8ANlaV\na2Pmkee+Dc97O577dgZy7hdqbmmn42YskiRJkjQwFnqSJEmSNDAWetJ4eBD4UP9V88tz34bnvR3P\nfTuee0mPcjMWSZIkSRoYO3qSJEmSNDAWepIkSZI0MBZ6kiRJkjQwFnqSJEmSNDAWetKYSXJmkv9O\n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\nJ0mSJEkD8/+dTM0zIUAqgQAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x164c7940>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plotting the boundaries\n",
    "plot_decision_regions(X_combined_std, y_combined, classifier = lr, test_idx=range(105, 150))\n",
    "plt.title('Plot of the test samples with the decision boundary when trained with index 5 and index 13 parameters.')\n",
    "plt.xlabel('Prefix_Suffix')\n",
    "plt.ylabel('URL_of_Anchor')\n",
    "plt.legend(loc='upper left')\n",
    "#plt.savefig('phishing_image_region.png', dpi = 300)\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python [Anaconda2]",
   "language": "python",
   "name": "Python [Anaconda2]"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
