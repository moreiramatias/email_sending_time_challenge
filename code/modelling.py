import numpy as np
import pandas as pd
import random
import math

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import lightgbm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

#reproducible seeds
seed_constant = 33
np.random.seed(seed_constant)
random.seed(seed_constant)

#my path
path = '/home/luismoreira/Desktop/pair_finance/'

#reading files from EDA
training_data = pd.read_csv(path + "proc.data/training_data.csv")
test_data = pd.read_csv(path + "proc.data/test_data.csv")


############################
## Modelling Stage
############################

#EVALUATION
#
#We start this part by defining an evaluation criteria to our predictions
#
#the idea is to minimize the difference between sent-emails and observed e-mails
#to do that, we compute the residuals to all possible observation times existing on the test-set
#and we select the minimum time.
#
#the residuals are corrected for the next day issue by adding 24h to the negative ones
#
#besides the desired metric to minimize (mae), the function also reports rmse and the percentages of cases
#that are read within 1h, 2h or in more than 8h
def evaluation(preds, obs, data, verbose = True):
    
    #give 15 minutes slack on predictions
    preds = preds - (15)
    residuals = obs - preds
    
    #correction of negative residuals to next day
    sel_idx = np.where(residuals < 0)[0]
    residuals[sel_idx] = residuals[sel_idx] + 24*60
    
    #select the closer observation time of each customer
    data = data.filter(['clientID']).copy()
    data['residuals'] = residuals
    residuals_df = data.groupby(['clientID']).min()
    residuals = residuals_df.residuals.to_numpy()
    
    #computing evaluation metrics
    rmse = math.sqrt(np.mean(residuals * residuals))
    mae = np.mean(residuals)
    perc_60 = len(np.where(residuals < 60)[0]) / len(residuals) * 100
    perc_120 = len(np.where(residuals < 120)[0]) / len(residuals) * 100
    perc_480 = len(np.where(residuals > 480)[0]) / len(residuals) * 100
    
    if (verbose):
        print(f"RMSE: {rmse:.2f}, MAE:{mae:.2f}, <1h:{perc_60:.2f}%, <2h:{perc_120:.2f}%, >8h:{perc_480:.2f}%")
    
    return residuals_df



#Model 0 - BASELINE
#
#we start modelling by creating a naive baseline
#the baseline simply generates random sending times from gaussian distributions
#which parameters come from the univariate empirical means/sd (observed) of the target

#baseline generation function
def baseline_generator(N, prob, mu_m, mu_e, sigma_m, sigma_e):
    
    #randomly deciding morning and afternoon peaks
    s = np.random.binomial(1, prob, N)
    
    # -1 for mornings, -2 for evening
    s = s-2
    
    #sampling times for morning peak
    n = len(np.where(s==-1)[0])
    np.random.seed(seed_constant)
    mp = np.random.normal(mu_m, sigma_m , n)
    
    #truncating values to 0 to 24*60
    sel_idx = np.where(mp<0)[0]
    if (len(sel_idx) > 0):
        mp[sel_idx] = 0
    
    sel_idx = np.where(mp>=24*60)[0]
    if (len(sel_idx) > 0):
        mp[sel_idx] = 0
    
    #sampling times for evening peak
    n = len(np.where(s==-2)[0])
    np.random.seed(seed_constant)
    ep = np.random.normal(mu_e, sigma_e, n)
    
    #truncating values to 0 to 24*60
    sel_idx = np.where(ep<0)[0]
    if (len(sel_idx) > 0):
        ep[sel_idx] = 0
    
    sel_idx = np.where(ep>=24*60)[0]
    if (len(sel_idx) > 0):
        ep[sel_idx] = 0
    
    #constructing final response
    sel_idx = np.where(s==-1)[0]
    s[sel_idx] = mp
    
    sel_idx = np.where(s==-2)[0]
    s[sel_idx] = ep
    
    return s



#estimating the probability of observed time comes from morning peak 
morning_peak = len(np.where(training_data.email_proc_obsblock == 1)[0])
afternoon_peak = len(np.where(training_data.email_proc_obsblock == 2)[0])
prob_morning = morning_peak / (morning_peak+afternoon_peak)

#getting baseline predictions
preds = baseline_generator(N = len(test_data), prob = prob_morning, mu_m = 9*60, sigma_m = 2*60, mu_e = 20*60, sigma_e = 3*60)
obs = test_data.email_proc_obstime

#showing evaluation of baseline
print("Baseline:")
_ = evaluation(preds, obs, test_data)


#these are the results we need to beat
#
#let's start by identifying fast responders as they seem pretty easy to model
#
#from the EDA section, we take a dependency on X1 and X3 (be 0/3)
#
#let's try to model that relationship linearly
training_model0 = training_data.filter(['clientID','client_X1','client_X3','email_proc_fast_responder']).drop_duplicates().copy()

                                       
#following observed pattern during EDA, we binarize client_X3 for this task
training_model0['client_X3_proc'] = 0
sel_idx = np.where((training_model0.client_X3 == 3) | (training_model0.client_X3 == 0))[0]
sel_idx = training_model0.index[sel_idx]
training_model0.loc[sel_idx,'client_X3_proc'] = 1
training_model0 = training_model0.filter(['client_X1','client_X3_proc','email_proc_fast_responder'])
training_model0.head()

#prepare test_set similarly
test_model0 = test_data.filter(['clientID','client_X1','client_X3','email_proc_fast_responder']).drop_duplicates().copy()
                                       
test_model0['client_X3_proc'] = 0
sel_idx = np.where((test_model0.client_X3 == 3) | (test_model0.client_X3 == 0))[0]
sel_idx = test_model0.index[sel_idx]
test_model0.loc[sel_idx,'client_X3_proc'] = 1
test_model0 = test_model0.filter(['client_X1','client_X3_proc','email_proc_fast_responder'])
test_model0.head()                                       
                                       
#modelling the probability of being a fast responder using logistic regression
RLR = LogisticRegression(penalty = 'none', 
                         solver = 'saga', 
                         random_state = seed_constant)

RLR.fit(training_model0.filter(['client_X1','client_X3_proc']),training_model0.email_proc_fast_responder)

#briefly inspecting resulting model        
dValues = RLR.coef_[0]
print('brief model view:')
print(len(dValues))
print(dValues)


#evaluating in the training set
outputs = RLR.predict(training_model0.filter(['client_X1','client_X3_proc']))
preds = RLR.predict_proba(training_model0.filter(['client_X1','client_X3_proc']))
preds_flat = pd.DataFrame(preds)[[1]]
roc_train = roc_auc_score(training_model0.email_proc_fast_responder, preds_flat)
print("Train AUC:")
print(roc_train)

acc = len(np.where(outputs == training_model0.email_proc_fast_responder)[0]) / len(outputs)
print("Train Accuracy:")
print(acc)

#evaluating in the test set
outputs = RLR.predict(test_model0.filter(['client_X1','client_X3_proc']))
preds = RLR.predict_proba(test_model0.filter(['client_X1','client_X3_proc']))
preds_flat = pd.DataFrame(preds)[[1]]
roc_test = roc_auc_score(test_model0.email_proc_fast_responder, preds_flat)
print("Test AUC:")
print(roc_test)

acc = len(np.where(outputs == test_model0.email_proc_fast_responder)[0]) / len(outputs)
print("Test Accuracy:")
print(acc)

#the model is not only strong but it generalizes well on unseen data
#let's store its outputs in the original dataset for further usage
#the idea is to use this as a maeta-feature to other models that may not be able
#to capture the linear effects of this relationship so well...
#

#adding processed feature X3 to original datasets

training_data['client_X3_proc'] = 0
sel_idx = np.where((training_data.client_X3 == 3) | (training_data.client_X3 == 0))[0]
sel_idx = training_data.index[sel_idx]
training_data.loc[sel_idx,'client_X3_proc'] = 1

test_data['client_X3_proc'] = 0
sel_idx = np.where((test_data.client_X3 == 3) | (test_data.client_X3 == 0))[0]
sel_idx = test_data.index[sel_idx]
test_data.loc[sel_idx,'client_X3_proc'] = 1

#adding predicted class and probabilities (meta-features) to original dataset
training_data['pe_email_proc_fast_responder'] = RLR.predict(training_data.filter(['client_X1','client_X3_proc']))
preds = RLR.predict_proba(training_data.filter(['client_X1','client_X3_proc']))
preds_flat = pd.DataFrame(preds)[[1]]
training_data['pp_email_proc_fast_responder'] = preds_flat

test_data['pe_email_proc_fast_responder'] = RLR.predict(test_data.filter(['client_X1','client_X3_proc']))
preds = RLR.predict_proba(test_data.filter(['client_X1','client_X3_proc']))
preds_flat = pd.DataFrame(preds)[[1]]
test_data['pp_email_proc_fast_responder'] = preds_flat

#
#MODEL 1 - VANILLA REGRESSION w/RF
#here, we will try simply to model the problem as a regression one
#
#one row per sent/obs. email, using all vanilla client features + prob. meta-feature
#please note that we tune RF to randomly select a splitting feature in each node
#and we reduce the size of bootstraps to 10%
#
#this aims to increase the heterogeneousity of each tree as we only have 4 features but 100s of rows
#

#learning model
max_samples_boots = int(0.1 * len(training_data))
model1 = RandomForestRegressor(n_estimators = 100, random_state = seed_constant, min_samples_split = 100, max_features = 1, max_samples = max_samples_boots)
_ = model1.fit(training_data.filter(['client_X1','client_X2','client_X3', 'pp_email_proc_fast_responder']), training_data.email_proc_obstime)

#getting predictions
preds = model1.predict(test_data.filter(['client_X1','client_X2','client_X3', 'pp_email_proc_fast_responder']))
obs = test_data.email_proc_obstime

#evaluating model in the test set
print("Model 1, Vanilla RF for regression:")
_ = evaluation(preds, obs, test_data)

#this solution is even worst than the baseline
#
#there are two problems: the loss function is symmetric (ours is not)
#On the top of that, the aggregation function in the leaves is the average...which severely contributes
#to a bias against larger observed times (afternoon)...aggravating further the abovementioned
#loss problem


#MODEL 2 - binary classification with RF, removing block 0 examples
#the idea with this model is to simply decide per customer
#if we should send e-mails in the morning or in the afternoon
#
#additionally, we omit evening examples as they are very few anyway
#and do not seem to follow any pattern

#remove time block 0 examples from training
training_data_m2 = training_data.copy()
sel_idx = np.where(training_data.email_proc_obsblock > 0)[0]
sel_idx = training_data_m2.index[sel_idx]
training_data_m2 = training_data_m2.iloc[sel_idx,:].copy()
training_data_m2.reset_index(drop = True,  inplace = True)

#learning classification model
max_samples_boots = int(0.1 * len(training_data_m2))
model2 = RandomForestClassifier(n_estimators = 100, random_state = seed_constant, min_samples_split = 100, max_features = 1, max_samples = max_samples_boots, oob_score = True)
_ = model2.fit(training_data_m2.filter(['client_X1','client_X2','client_X3','pp_email_proc_fast_responder']), training_data_m2.email_proc_obsblock)

#inspecting OOB accuracy to check how good the model discriminates both classes
print("OOB Accuracy")
print(model2.oob_score_)

#preparing prediction from classification labels
#we use a conservative approach to the estimation of the predictions
#by taking two fixed times to send emails: the beginning of the intervals
preds = model2.predict(test_data.filter(['client_X1','client_X2','client_X3','pp_email_proc_fast_responder']))
preds[np.where(preds == 2)] = 13*60+15
preds[np.where(preds == 1)] = 7*60+15

obs = test_data.email_proc_obstime

print("Model 2, Vanilla RF for bin. classification:")
_ = evaluation(preds, obs, test_data)

#this model is already slightly better than our baseline...uff, we did it!
#
#however, we are still working with a symmetric loss 
#moreover, the affected intervals are very lengthy

#
#
#MODEL 3 - lightGBM w/ custom loss regression
#
#the tradeoff between the non-linearities of the target function that we are trying to approximate
#vs. the optimization procedures, loss functions and ensemble aggregations
#that come on off-the-shelf learners (like the ones we are using here) is challenging.
#
#One of the main modelling problems are the "next day" predictions...and the associated assymetric errors.
#
#if the algorithm "risks" on converging to send an e-mail late in the day,
#the chance that it will get read next day increases massively - as the e-mails read during nighttime are neglectible.
#that turns that the loss of missing the customer's reading time is not -1h but +10h instead (if not more).
#
#this problem has an assimetric nature and to model it more adequate we will need
#an an assymetric loss. 
#
#we will do it with lightGBM package, which supports user-defined losses
#as long as the hessian and the gradient are defined (mandatory for gbm-like algorithms).
#
#the main reason to prefer to try regression vs. classification w/custom-loss is on the metric that is desired to optimize:
#the difference between sent e-mail time and obs. e-mail time.
#


#
#
#processing data
training_data_m4 = training_data.copy()
test_data_m4 = test_data.copy()

#remove evening examples from training
#filtering out cases only during the night
sel_idx = np.where((training_data_m4.email_proc_obshour >= 7) & (training_data_m4.email_proc_obshour < 24))[0]
sel_idx = training_data_m4.index[sel_idx]
training_data_m4 = training_data_m4.iloc[sel_idx,:].copy()
training_data_m4.reset_index(drop = True, inplace = True)

#preparing to learn model
training_model4 = training_data_m4.filter(['client_X2', 'pp_email_proc_fast_responder'])
training_label4 = training_data_m4.email_proc_obstime

test_model4 = test_data_m4.filter(['client_X2', 'pp_email_proc_fast_responder'])
test_label4 = test_data_m4.email_proc_obstime

#custom assymetric loss
#it is difficult to add a constant to the residuals
#due to the fact that we need both the gradient and the hessian to use GBM
#
#instead, we will use a linear penalty factor of 100
#to force clearly the algorithm to correct the error on these examples...
def custom_asymmetric_train(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")
    
    #assymetric penalty for negative errors that pass for next day
    penalty = 100
    
    #rough calculation of gradient and hessian
    grad = np.where(residual<0, -2*penalty*residual, -2*residual)
    hess = np.where(residual<0, 2*penalty, 2.0)
    return grad, hess


#here we learn the model with very standard hyperparameters + our custom loss
#with exception of the subsample at 10%, which is similar to what we already
#did with RF for the very same reasons
model4 = lightgbm.LGBMRegressor() 

model4.set_params(**{'objective': custom_asymmetric_train}, 
                  learning_rate=0.3, 
                  n_estimators = 10, 
                  subsample = 0.1,
                  random_state = seed_constant)


model4.fit( X = training_model4, y = training_label4)

#getting predictions
preds = model4.predict(test_model4)
obs = test_label4

#evaluating the model
print("Model 3, lightgbm with assymetric custom loss for regression:")
residuals = evaluation(preds, obs, test_data_m4)

#this model is a considerable improvement to anything above!
#in average, a customer takes ~2h to open the sent e-mails based on our model.
#52% of the e-mails are read in 1h and 73% are read within 2h. Only 6% take more than 8h to read.
#
#

#How to improve?#
#let's look to the distribution of our predictions...
plt.figure(figsize=(24,10))
sns.set(font_scale=3)
sns.distplot(preds/60, kde=False, color='red', bins = 24, norm_hist = True)
#as you can see, this distribution is very far from the original target one
#does this makes sense?
#
#Yes, it does. If we look again carefully to the different densities observed in EDA vs. different values of 
#these two variables, we see that the users either read the e-mail at morning peak
#at evening peak or both.
#
#a conservative approach to minimize large errors (the next day issue) would be to always send e-mails during the morning peak and optimize 
#the sending times in that period...as users always tend to always have at least one observed time there.
#
#consequently, the cost-sensitive nature of this learner forces it to not take chances on predicting even on sending e-mails after 10h,
#concentrating all the sendings between 8h-10h.

#from EDA, and among the available features, it seems not to exist much more predict power to do significantly better than this.


#
#
#naturally, one way to improve is to be able to safelly send e-mails later in the day,
#enlarging the prediction outputs. This learner also goes after conditional means
#which...in turn, are much more frequent in the morning...this, together with the defined loss,
#results in the present outcome.
#
#
#To counter this "safe conditional mean" effect, I would explore further three possible directions:
#
#
#1) Quantile Regression: I would explore quantile regression to create p
#Prediction intervals around which I would decide to risk more or less, 
#depending on their center and width. However, the present features may not allow to get significant improvements from this.
#
#
#2) another possibility would be to explore further cost-sensitive learning
#besides the custom loss, we could try to balance the under-representativeness of some
#data points  - which can also be contributing for this "safe conditional mean" effect
#
#
#3) finally, I would tune xgboost parameters. That would include also the loss penalty.
#a smart way to explore the solution space could be either BO or random search with 5-fold CV over training 
#whereas prediction error would be measured on a custom validation function (a wrapper over our evaluation one)
#
#the main advantage of this solution is to take out the tuning to an outer optimization loop
#where we are not constrained to the mathematical needs of the GBMs...and thus we can optimize our MAE directly.
