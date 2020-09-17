# The "E-Mail Sending Time" Challenge

**Author: Luis Moreira-Matias (luis.moreira.matias in Gmail)**

**Date: 10/09/2020**

Given a dataset (**orig.data folder**) with users features and emails sent/open times: N users (N=100.000), every user data lays in separate row (i.e. you have 100K rows for 100K users).

### Feature Descrption

X[i] - features of i-th user, i in [1..N]
  X[i][0] - feature 1 (numerical)
  X[i][1] - feature 2 (numerical)
  X[i][2] - feature 3 (categorical)

M[i] - number of emails sent to i-th user (1 <= M[i] <= 15), Mi is not the same for every user i,
(for instance M[1] could be 10, M[2] could be 5)
  TS[i][j] - time of j-th email sending to i-th user, j in [1..M[i]]
  TO[i][j] - time of j-th email opening by i-th user, j in [1..M[i]]

**NOTE**: If TS[i][j] > TO[i][j], then TO[i][j] belongs to the next day (for instance if sending time is 14:00 and opening time is 10:00, then it’s 10:00 of the next day)

**Goal**:​ given features of new user X, predict the sending time TS so that opening time TO will
be as early as possible, i.e (TO-TS) -> min

Hint: ​ Users can open emails immediately or some time during a day (like some users prefer
to read their email after lunch, some of them do it mornings, some do both)

**Task Description**: Build and train your model, discussing on how it could be improved in the future.

## Executive Summary of Solution

The proposed solution is delivered under two chapters: I) Exploratory Data Analysis and II) Modelling.


* **Input data**: 100k customers with 3 features (X1,X2: continuous, X3:categorical). Each customer was sent 1-15 e-mails. The sent/reading e-mail times are also given.
* **Challenge**: Build a model that is able to predict the possible the best e-mail sending time to each customer, minimizing the elapsed time to its reading time.

* **EDA**: X1, X2 and X3 seem to be fairly independent among themselves;
* **EDA**: For 20% of the e-mails sent, the customers seem to respond immediately (<45m);
* **EDA**: Input data seems to be synthetically generated;
* **EDA**: Customers read most of the e-mails in the evening and morning peaks (around 20h and 09h), respectively. The e-mail readings almost do not happen during nighttime (00h-07h).
* **EDA**: The number of e-mails sent to each customer as well as the sending times seem to be drawn at random;
* **EDA**: X1 and X3 have a nearly linear dependence on predicting if a customer is a fast responder or not;
* **EDA**: X2 has a dependence on the observed time;

* **Modelling**: Major modelling problem is the assymetric nature of our optimization metric as, if we predict a sending time that occurs after the optimal reading time, we need to assume that the e-mail will only be read on the next day (i.e. a prediction error of -60m is actually an error of 23h!);
* **Solution**: GBM (w/ lightGBM package) + a customized loss function (assymetric). It uses two features: 1) X2 and a 2) linear combination of X1 and X3. 
* **Result**: Average reading time of 2h (120m). In test set, 50% of e-mails sent are read within 1h while 75% are read within 2h.


## Reproducibility Issues

* If there is interest to recompile the code/notebooks, it is necessary to the manual assign of an absolute path inside of the code - that can be found right after library imports. 

* Ideally, the code would only reproduce perfectly inside of a VM/docker. In the absence of that, I report on the major versions and packages used to create this.:

  * Ubuntu 18.04
  * Python 3.6.9
  * Numpy 1.18.2
  * Pandas 1.0.3
  * matplotlib 3.2.1
  * Seaborn 0.10
  * Sklearn 0.22.2.post1
  * lightgbm 3.0.0
  * jupyter core: 4.6.3
  * jupyter-notebook: 6.0.3


Please proceed on [this link](notebook/challenge_solution.ipynb) to explore my solution. If you prefer good old HTML, please follow [this link](START_HERE.html) instead.
