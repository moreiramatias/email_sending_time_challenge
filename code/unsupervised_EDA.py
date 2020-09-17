import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math

from grepfunc import grep



#reproducible seeds
seed_constant = 33
np.random.seed(seed_constant)
random.seed(seed_constant)

#my path
path = '/home/luismoreira/Desktop/email_sending_time_challenge/'

#reading clientdata
data = pd.read_csv(path + "orig.data/data.csv")
data.info()
data.head()

#let's separate client features from e-mail activity/labels, adding a primary key beforehand
#let's also rename the features with prefixes for easier identification
clientdata = data.filter(['X1','X2','X3']).copy()
clientdata = clientdata.add_prefix('client_')
emaildata = data.drop(columns = ['X1','X2','X3']).copy()
emaildata = emaildata.add_prefix('email_')

clientdata['clientID'] = np.arange(0,len(clientdata)) + 1
emaildata['clientID'] = np.arange(0,len(emaildata)) + 1
data = clientdata.merge(emaildata, on = 'clientID').copy().reset_index()

#sanity check...
data.head()
data.columns

#let's focus on clientdata for the timebeing...
#We know we have 3 features...X1,X2 are numerical while X3 is said to be categorical
#
#let's try to understand if we have missing values in any of the features
#
NA_cols = clientdata.columns[clientdata.isna().any()]
print(len(NA_cols))

#no missing values. Let's now look to the univariate distributions of each feature
#

clientdata.drop(columns =['clientID']).hist(figsize=(24,24))

#X1 and X2 have a somewhat gaussian distribution while X3 follows an uniform one.
#The levels of X3 are ordinal...however, we do not know if that order actually exists.
#For the sake of time, We will take the information given by the assessment as ground truth and ignore any ordinality that may exist in X3.
# 
#let's try to understand how they relate with each other
g = sns.pairplot(clientdata.drop(columns =['clientID']), hue="client_X3", palette="Set2", diag_kind="kde", height=5, corner=True, markers="+", diag_kws=dict(shade=True))

#X1 and X2 seem to be fairly independent of X3 as their distributions among the different classes are very similar
#However, p(X1,X2|X3) seems to change for different X3 levels - in specific, it is plausible that the variance of p(X1,X2) is slightly larger for X3=0 than when X3=3.
#
#There is not much that can be concluded from these analysis that could be taken as an actionable insight.
#
#
#SUPERVISED EDA
#

#
#separate train and test to avoid data leakage about the labels

#shuffle order to avoid input biases
data = data.reindex(np.random.permutation(data.index)).copy()
data.reset_index(drop = True, inplace = True)
data.head()

#randomly sampling 70% of the data for training and 30% for testing

#70% for training
training_data = data.sample(n=math.floor(0.7 * len(data.index)), random_state = seed_constant).copy()
training_data.reset_index(drop = True, inplace = True)
training_data.drop(columns = 'index', inplace = True)

training_IDs = pd.Series(training_data.clientID)
idx_test = np.where(np.logical_not(data.clientID.isin(training_data.clientID)))[0]
idx_test = data.index[idx_test]

#30% for testing
test_data = data.iloc[idx_test,:].copy()
test_data.reset_index(drop = True, inplace = True)
test_data.drop(columns = 'index', inplace = True)

#sanity check
print(training_data.shape)
print(test_data.shape)
training_data.head()
test_data.head()
print(training_data.clientID.isin(test_data.clientID).to_numpy().any())
#done!!

#let's start by observing the distribution of e-mails sent to each customer
training_data.email_M.hist(bins = 15, figsize=(24,10))
#it seems to be uniform. will it be related to the typical e-mail obser. time of each customer?


#In order to answer that question, analyze e-mail sent observed data better
#and its potential relationship with clientdata features
#we would need to "melt" the dataset, transforming each row into a record
#of an e-mail/sent and observed
#

#melting dataset
feats_to_melt = grep(training_data.columns, "email_TS", i=False, v=False)
training_data_melted_ts = pd.melt(training_data, id_vars = 'clientID', value_vars = feats_to_melt)

feats_to_melt = grep(training_data.columns, "email_TO", i=False, v=False)
training_data_melted_to = pd.melt(training_data, id_vars = 'clientID', value_vars = feats_to_melt)

feats_to_keep = grep(training_data.columns, "email_T", i=False, v=True)
training_data_melted = training_data_melted_ts.merge(training_data.filter(feats_to_keep), how="left", left_on="clientID", right_on = 'clientID').copy().reset_index()
training_data_melted = training_data_melted.merge(training_data_melted_to, on="clientID").copy().reset_index()
training_data_melted.drop(columns = ['level_0','index'], inplace = True)

#filtering only correct and relevant lines/combinations
sel_cases_idx = np.where(training_data_melted.variable_x.str.slice(start = 8, stop = 10) == training_data_melted.variable_y.str.slice(start = 8, stop = 10))[0]
sel_cases_idx = training_data_melted.index[sel_cases_idx]

training_data_melted = training_data_melted.iloc[sel_cases_idx,:].copy()
training_data_melted.reset_index(drop = True, inplace = True)

sel_cases_idx = np.where(np.logical_not(training_data_melted.isnull().any(axis=1)))[0]
sel_cases_idx = training_data_melted.index[sel_cases_idx]

training_data_melted = training_data_melted.iloc[sel_cases_idx,:].copy()
training_data_melted.reset_index(drop = True, inplace = True)



#sanity check
training_data_melted.head(50)
training_data_melted.shape
#done


#now, we need to transform these sending/observed times into greatnesses that we can work with
#To do it so, we will create three new processed variables:
#1) response time (sent-obs in min.)
#2) e-mail observed time-of-the-day (minutes)
#3) e-mail observed time-of-the-day discrete  (hours)


#minutes sent
hours = training_data_melted.value_x.str.slice(start = 0, stop = 2).to_numpy()
minutes = training_data_melted.value_x.str.slice(start = 3, stop = 5).to_numpy()
training_data_melted['email_senttime'] = hours.astype(int) * 60 + minutes.astype(int)

#minutes observed
hours = training_data_melted.value_y.str.slice(start = 0, stop = 2).to_numpy()
minutes = training_data_melted.value_y.str.slice(start = 3, stop = 5).to_numpy()
training_data_melted['email_proc_obstime'] = hours.astype(int) * 60 + minutes.astype(int)

#hours observed
training_data_melted['email_proc_obshour'] = hours.astype(int)

#response_time
training_data_melted['email_proc_restime'] = training_data_melted['email_proc_obstime'] - training_data_melted['email_senttime']
sel_cases_idx = np.where(training_data_melted['email_proc_restime'] < 0)[0]
sel_cases_idx = training_data_melted.index[sel_cases_idx]
training_data_melted.loc[sel_cases_idx,'email_proc_restime'] = training_data_melted.loc[sel_cases_idx,'email_proc_restime'] + (24*60)


#dropping original melted columns
training_data_melted = training_data_melted.drop(columns = ['variable_x','value_x', 'variable_y', 'value_y'])

#sanity check
training_data_melted.head(50)
training_data_melted.shape
#done

#let's quickly check if all proc. observed times are valid before proceeding
df = training_data_melted['email_proc_obshour'].value_counts()
print(df)

#there is a strange negative number...let's see if it also exists 
#sometehing problematic on sent times
print(len(np.where((training_data_melted.email_senttime < 0) | (training_data_melted.email_senttime >= 24*60))[0]))

#no...so, let's just remove the strange and isolated example from training before proceeding
sel_cases_idx = np.where(training_data_melted.email_proc_obshour >= 0)[0]
sel_cases_idx = training_data_melted.index[sel_cases_idx]

training_data_melted = training_data_melted.iloc[sel_cases_idx,:].copy()
training_data_melted.reset_index(drop = True, inplace = True)



#quickly analyzing response time 
plt.figure(figsize=(24,10))
sns.set(font_scale=3)
ab = sns.distplot(training_data_melted['email_proc_restime'], kde=False, color='red', norm_hist = True)

#we notice that, in ~20% of the emails sent, the customers respond fast whereas then an uniform pattern is followed.
#let's use this value to define a 'fast response customer'.
#
#to do that, let's start by inspecting the lowest 20% values...which correspond to the first 3 bins in the histogram
min_th = training_data_melted['email_proc_restime'].quantile(0.20)
sel_cases_idx = np.where(training_data_melted['email_proc_restime'] < min_th)[0]
sel_cases_idx = training_data_melted.index[sel_cases_idx]

plt.figure(figsize=(24,10))
sns.set(font_scale=3)
ab = sns.distplot(training_data_melted.loc[sel_cases_idx,'email_proc_restime'], kde=False, color='red', norm_hist = True)
print(min_th)

#the distribution is actually quite uniformily descending without a particular spike within
#this turns the selected threshold (20% quantile, 43min. of response time) quite realistic to assume an immediate reading of the e-mail
#

#there is also a particularly funny effect: there are no response times equal to multiples of 7
#as it is easily confirmable analytically
print(ab.patches[7-1].get_height())
print(ab.patches[14-1].get_height())
print(ab.patches[14*3-1].get_height())
#this effect points out that this data is not real but synthetically generated instead.

#let's now quickly analyze hour of the day (observed)
plt.figure(figsize=(24,10))
sns.set(font_scale=3)
sns.distplot(training_data_melted['email_proc_obshour'], kde=False, color='red', bins = 24, norm_hist = True)
#here we notice that the customers barely read e-mails from 0 to 6am, showing two peaks around 09h and 20h (as observed before)
#we may also conclude that it is not worthy to send e-mails between 0-6am at all.
#

#let's see if exists any relationship between this variable and the number of e-mails sent to each customer...
#if that exists, we may generate some aggregation features of these numbers vs. clientdata...
data_viz = training_data_melted.filter(['email_proc_obshour', 'email_M']) 
plt.figure(figsize=(24,10))
sns.set(font_scale=3)
g = sns.boxplot(data = data_viz, hue="email_M", palette="pastel",   x = 'email_M', y = 'email_proc_obshour')
g.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), ncol=1)
#no...the number of e-mails sent to each customer seems to be independent from the typical e-mail observed time


#
#let's discretize the reading time in three folds: 1: 07h-13h, 2: 13h-24h, 0: other time periods of the day
#
#the intuition to use these bins is the peaks/valleys observed in the figure + the hint on the assessment description
#about the customers observe the e-mail before/after lunch (defined here as 13h)
#


training_data_melted['email_proc_obsblock'] = 0

#7h-13h
sel_idx = np.where((training_data_melted['email_proc_obshour'] >= 7) & (training_data_melted['email_proc_obshour'] < 13))[0]
sel_idx = training_data_melted.index[sel_idx]
training_data_melted.loc[sel_idx,'email_proc_obsblock'] = 1
print(len(sel_idx))

#13h-24h
sel_idx = np.where((training_data_melted['email_proc_obshour'] >= 13) & (training_data_melted['email_proc_obshour'] < 24))[0]
sel_idx = training_data_melted.index[sel_idx]
training_data_melted.loc[sel_idx,'email_proc_obsblock'] = 2
print(len(sel_idx))


#Now that we have some new e-mail-related processed variables, let's use pairplots explore relationships between these 
#new variables, the number of messages sent to each customer and clientdata
#
#
data_viz = training_data_melted.filter(['email_proc_obsblock','client_X1','client_X2','client_X3', 'email_proc_restime'])                                           
g = sns.pairplot(data_viz,  hue="email_proc_obsblock", palette="pastel",  height = 6, corner=True, markers="+")
   

#there are 2 conclusions to take out of this chart
# 1) X1, X2 and X3 seem to have influence over the observed e-mail time
# 2) The response time seems invariably different across time blocks
#
# let's now look more closely on the response times

data_viz = training_data_melted.filter(['email_proc_obsblock', 'email_proc_restime']) 
plt.figure(figsize=(24,10))
sns.set(font_scale=3)
g = sns.boxplot(data = data_viz, hue="email_proc_obsblock", palette="Set2",   x = 'email_proc_obsblock', y = 'email_proc_restime')
g.legend(loc='center left', bbox_to_anchor=(1., 0.5), ncol=1)
plt.show()

# The e-mail opening time seem to be lower whenever it is received on time block 2
# This may be explained if the e-mail sent times suffer some bias (namely, if we are sending more e-mails at these periods than throughout the day)
#
# let's quickly assess that by looking to the distribution of sent times
plt.figure(figsize=(24,10))
sns.set(font_scale=3)
ab = sns.distplot(training_data_melted['email_senttime'], kde=False, color='red', norm_hist = True)

#indeed, the distribution of email sending times is nearly uniform...consequently, whomever reads the e-mails after 14h-00h tends to read them faster
#does this makes sense? not really.
#
#This observed "pattern" in the data can also be explained if people just reads the e-mails in certain periods of the day... pushing e-mails sent in other times to be read in the next day.
#This behavior will enlarge considerably the reading time
#
#let's try to observe that in the data by 1) excluding those 'next day' reading points and 2) redo the boxplots
#if a pattern of "users that read their e-mails later in the day have shorter reading times", this must be observable in this procedure.
#

#filtering 'only same day' cases
training_data_melted['email_proc_restime_special'] = training_data_melted['email_proc_obstime'] - training_data_melted['email_senttime']
sel_cases_idx = np.where(training_data_melted['email_proc_restime_special'] >= 0)[0]
sel_cases_idx = training_data_melted.index[sel_cases_idx]
data_viz = training_data_melted.iloc[sel_cases_idx,:].copy()
data_viz.reset_index(drop = True,  inplace = True)

#replotting the boxplots of response times
plt.figure(figsize=(24,10))
sns.set(font_scale=3)
g = sns.boxplot(data = data_viz, hue="email_proc_obsblock", palette="Set2",   x = 'email_proc_obsblock', y = 'email_proc_restime_special')
g.legend(loc='center left', bbox_to_anchor=(1., 0.5), ncol=1)
plt.show()
#in this chart, we depict that the reading times are actually longer in the afternoon than in the morning
#this makes sense as the block 13h-24h is way larger (50%) than the block 07h-13h and as we excluded "next day cases"
#
#This is also a good hint as there no trend on the response times whatsoever. 
#Most likely, the e-mails are sent completely at random.
#
#

#
#LABEL CONSTRUCTION
#
#now we need to formulate our ML problem as a supervised learning one, focusing on the e-mail opening times.
#for that, we will need to create labels
#
# There are two main options to model the problem: 1) regression or 2) classification
#
#1) regression using a continuous target is an option
#however, the default loss functions are symmetric...penalizing equalliy errors regardless of their signal
#moreover, there are some customers that simply answer immediately to their e-mails...adding noise to the modelling that needs to be done for others
#

#2) classification with a discretized target
#would suffer the same problem than regression - common loss functions are symmetric
#Therefore, in order to achieve good results, we would need to express a quantity on the loss as different types of errors will cost more than others.
#

#to explore these options, we will create three labels:
#
#i) opening times (minutes), (continuous) - already there as "email_proc_obstime"
#ii) opening times (blocks). (discrete) - already there as "email_proc_obsblock"
#iii) respond immediately (binary) yes/no
#
#in order to compute the latter, we have to define a pattern behavior as we consider to respond immediately
#we will define it as follows: if a customer answered in 60 minutes (using the 43 minutes computed above as reference and giving some slack)
# to more than 50% of the sent e-mails in two different blocks of the day, he is considered to be an immediate responder.
# otherwise, he is considered a seasonal one.
#
#let's implement this.

data_viz = training_data_melted.filter(['clientID','email_proc_restime','email_proc_obstime', 'email_proc_obsblock'])
min_th_fast_response = 60

#create number of favorable cases vs. number of observations
data_viz2 = data_viz.copy()
sel_cases_idx = np.where(data_viz2.email_proc_restime < min_th_fast_response)[0]
sel_cases_idx = data_viz2.index[sel_cases_idx]
data_viz2 = data_viz2.iloc[sel_cases_idx,:].copy()
data_viz2.reset_index(drop = True,  inplace = True)

fast_counts = data_viz2.groupby(['clientID']).size().to_frame()
fast_counts.reset_index(level=0, inplace=True)
fast_counts = fast_counts.rename(columns={0: "fast_counts"})

all_counts = data_viz.groupby(['clientID']).size().to_frame()
all_counts.reset_index(level=0, inplace=True)
all_counts = all_counts.rename(columns={0: "all_counts"})

fast_counts = fast_counts.merge(all_counts, how="left", left_on="clientID", right_on = 'clientID').copy().reset_index(drop = True)
fast_counts['perc_fast_counts'] = fast_counts.fast_counts / fast_counts.all_counts

#selecting customers with 4+ e-mails sent/received and with 50%+ of fast responses
sel_fast_cases_idx = np.where((fast_counts.perc_fast_counts >= 0.5) & (fast_counts.all_counts >= 4))[0]
fast_counts = fast_counts.iloc[sel_fast_cases_idx,:].copy()
fast_counts.reset_index(drop = True, inplace = True)


#merge to prune out invalid clientIDs with sanity check
print(fast_counts.shape)
fast_counts = fast_counts.merge(data_viz2, how="left", left_on="clientID", right_on = 'clientID').copy().reset_index(drop = True)
print(len(fast_counts.clientID.unique()))
#done

#removing all non-fast cases w/ sanity check
print(len(fast_counts.clientID.unique()))
sel_cases_idx = np.where(fast_counts.email_proc_restime < min_th_fast_response)[0]
sel_cases_idx = fast_counts.index[sel_cases_idx]
fast_counts = fast_counts.iloc[sel_cases_idx,:].copy()
fast_counts.reset_index(drop = True,  inplace = True)
print(len(fast_counts.clientID.unique()))

#counting occurrences in different periods of the day
df = fast_counts.groupby('clientID')['email_proc_obsblock'].value_counts().unstack()
df = df.count(axis=1).to_frame()
df.reset_index(level=0, inplace=True)
df = df.rename(columns={0: "different_periods"})

#excluding cases which occur always in the same period of the day w/sanity check
print(df.shape)
sel_cases_idx = np.where(df.different_periods > 1)[0]
sel_cases_idx = df.index[sel_cases_idx]
df = df.iloc[sel_cases_idx,:].copy()
df.reset_index(drop = True,  inplace = True)
print(df.shape)

#adding info to original training set, w/ sanity check
print(df.shape)
training_data_melted['email_proc_fast_responder'] = 0
sel_cases_idx = np.where(training_data_melted.clientID.isin(df.clientID))[0]
sel_cases_idx = training_data_melted.index[sel_cases_idx]
training_data_melted.loc[sel_cases_idx,'email_proc_fast_responder'] = 1
print(len(training_data_melted.clientID[sel_cases_idx].unique()))

#let's explore again clientdata with this new label...
data_viz = training_data_melted.filter(['email_proc_fast_responder','client_X1','client_X2','client_X3'])                                           
g = sns.pairplot(data_viz, hue="email_proc_fast_responder", palette="Set2",  height=5, corner=True, markers="+")

#X1 and X3 have a very strong predictive power over this new label
#moreover, the relationship seems to be linear
#let's keep this in mind for the modelling stage
#
#

#let's now refine our exploratory data analysis by redoing the pairplot we've done above
#but now excluding cases from off peak periods (time block 0, afternoon and evening)
#as well as fast responders

sel_cases_idx = np.where((training_data_melted.email_proc_fast_responder == 0) & (training_data_melted.email_proc_obsblock > 0))[0]
sel_cases_idx = training_data_melted.index[sel_cases_idx]
data_viz = training_data_melted.iloc[sel_cases_idx,:].copy()
data_viz.reset_index(drop = True, inplace = True)
data_viz.head()

data_viz = data_viz.filter(['email_proc_obsblock','client_X1','client_X2','client_X3'])                                           
g = sns.pairplot(data_viz,  hue="email_proc_obsblock", palette="pastel",  height = 6, corner=True, markers="+")
#in this chart, it is clear that X2 has a obvious predictive power to distinguish between customers
#that prefer to read e-mails at evening vs. customers that prefer to read them in the mornings
#
#let's try to understand how much is this power by analyzing boxplots of our target variable 
#by the quantiles of client_X2 variable...
training_data_melted['client_X2_disc']  = pd.qcut(training_data_melted['client_X2'], q=4)
plt.figure(figsize=(24,10))
sns.set(font_scale=3)
g = sns.boxplot(data = training_data_melted,  palette="Set2",   x = 'client_X2_disc', y = 'email_proc_obshour')

#in this plot, X2 seem to be quite unrelated with our target in a continuous fashion
#however, there is a clearly mean shift for positive values of this variable
#
#let's try to see how the target densities shift based on this cutoff point of client_X2

training_data_melted['client_X2_pos']  = pd.qcut(training_data_melted['client_X2'], q=2)
data_viz = training_data_melted.filter(['email_proc_obshour','client_X2_pos'])
g = sns.pairplot(data_viz,  hue="client_X2_pos", palette="pastel",  height = 15, corner=True, markers="+")
g._legend.remove()
#for positive values of client X2, the users typically read the e-mails only in the morning
#for negative values of client_X2, the users either read the e-mails in the morning or in late afternoon
#


#
#
#let's now understand how the observed times distribution is affected by the removal of the "immediate readers" as well.
#
#to do that, we create a new feature w/ 4 levels 
#that basically maps all possible cases between fast_readers yes/no and client_X2_positive_value yes/no 
#
#we use that feature to colour the different densities
training_data_melted["client_X2_pos"] = training_data_melted["client_X2_pos"].astype('category')
training_data_melted["client_X2_pos"] = pd.Categorical(training_data_melted["client_X2_pos"]).rename_categories(['0','1'])
training_data_melted['new_category'] = np.array(list(map(int, training_data_melted["client_X2_pos"].to_numpy()))) * 2 + training_data_melted["email_proc_fast_responder"].to_numpy()

data_viz = training_data_melted.filter(['email_proc_obshour','new_category'])
g = sns.pairplot(data_viz,  hue='new_category', palette="Set2",  height = 15, corner=True, markers="+")
g._legend.remove()
#the distribution is indeed different for fast readers (regardless of the value of X2)...being uniform between 07h-20h
#
#these two variables have the most predictive power...let's keep that in mind for modelling stage as well.

#
#
#Now that we've completed EDA, let's prepare both the training set and the test set for the modelling stage
#
#
#columns to keep 
training_data_melted = training_data_melted.filter(['clientID', 'client_X1', 'client_X2', 'client_X3', 'email_proc_obstime', 'email_proc_obshour', 'email_proc_restime', 'email_proc_obsblock', 'email_proc_fast_responder','email_M'])

#apply processing to the test_set as well

#melting
feats_to_melt = grep(test_data.columns, "email_TS", i=False, v=False)
test_data_melted_ts = pd.melt(test_data, id_vars = 'clientID', value_vars = feats_to_melt)

feats_to_melt = grep(test_data.columns, "email_TO", i=False, v=False)
test_data_melted_to = pd.melt(test_data, id_vars = 'clientID', value_vars = feats_to_melt)

feats_to_keep = grep(test_data.columns, "email_T", i=False, v=True)
test_data_melted = test_data_melted_ts.merge(test_data.filter(feats_to_keep), how="left", left_on="clientID", right_on = 'clientID').copy().reset_index()
test_data_melted = test_data_melted.merge(test_data_melted_to, on="clientID").copy().reset_index()
test_data_melted.drop(columns = ['level_0','index'], inplace = True)

#filtering only relevant lines
sel_cases_idx = np.where(test_data_melted.variable_x.str.slice(start = 8, stop = 10) == test_data_melted.variable_y.str.slice(start = 8, stop = 10))[0]
sel_cases_idx = test_data_melted.index[sel_cases_idx]

test_data_melted = test_data_melted.iloc[sel_cases_idx,:].copy()
test_data_melted.reset_index(drop = True, inplace = True)

sel_cases_idx = np.where(np.logical_not(test_data_melted.isnull().any(axis=1)))[0]
sel_cases_idx = test_data_melted.index[sel_cases_idx]

test_data_melted = test_data_melted.iloc[sel_cases_idx,:].copy()
test_data_melted.reset_index(drop = True, inplace = True)


#sanity check
test_data_melted.head(50)
test_data_melted.shape


#construct features

#minutes sent
hours = test_data_melted.value_x.str.slice(start = 0, stop = 2).to_numpy()
minutes = test_data_melted.value_x.str.slice(start = 3, stop = 5).to_numpy()
test_data_melted['email_senttime'] = hours.astype(int) * 60 + minutes.astype(int)

#minutes observed
hours = test_data_melted.value_y.str.slice(start = 0, stop = 2).to_numpy()
minutes = test_data_melted.value_y.str.slice(start = 3, stop = 5).to_numpy()
test_data_melted['email_proc_obstime'] = hours.astype(int) * 60 + minutes.astype(int)

#hours observed
test_data_melted['email_proc_obshour'] = hours.astype(int)

#response_time
test_data_melted['email_proc_restime'] = test_data_melted['email_proc_obstime'] - test_data_melted['email_senttime']
sel_cases_idx = np.where(test_data_melted['email_proc_restime'] < 0)[0]
sel_cases_idx = test_data_melted.index[sel_cases_idx]
test_data_melted.loc[sel_cases_idx,'email_proc_restime'] = test_data_melted.loc[sel_cases_idx,'email_proc_restime'] + (24*60)


#dropping original melted columns
test_data_melted = test_data_melted.drop(columns = ['variable_x','value_x', 'variable_y', 'value_y'])

#discretization of the target
test_data_melted['email_proc_obsblock'] = 0

sel_idx = np.where((test_data_melted['email_proc_obshour'] >= 7) & (test_data_melted['email_proc_obshour'] < 13))[0]
sel_idx = test_data_melted.index[sel_idx]
test_data_melted.loc[sel_idx,'email_proc_obsblock'] = 1
print(len(sel_idx))

sel_idx = np.where((test_data_melted['email_proc_obshour'] >= 13) & (test_data_melted['email_proc_obshour'] < 24))[0]
sel_idx = test_data_melted.index[sel_idx]
test_data_melted.loc[sel_idx,'email_proc_obsblock'] = 2
print(len(sel_idx))


#adding up the fast email flag
data_viz = test_data_melted.filter(['clientID','email_proc_restime','email_proc_obstime', 'email_proc_obsblock'])


#create number of favorable cases vs. number of observations
data_viz2 = data_viz.copy()
sel_cases_idx = np.where(data_viz2.email_proc_restime < min_th_fast_response)[0]
sel_cases_idx = data_viz2.index[sel_cases_idx]
data_viz2 = data_viz2.iloc[sel_cases_idx,:].copy()
data_viz2.reset_index(drop = True,  inplace = True)

fast_counts = data_viz2.groupby(['clientID']).size().to_frame()
fast_counts.reset_index(level=0, inplace=True)
fast_counts = fast_counts.rename(columns={0: "fast_counts"})

all_counts = data_viz.groupby(['clientID']).size().to_frame()
all_counts.reset_index(level=0, inplace=True)
all_counts = all_counts.rename(columns={0: "all_counts"})

fast_counts = fast_counts.merge(all_counts, how="left", left_on="clientID", right_on = 'clientID').copy().reset_index(drop = True)
fast_counts['perc_fast_counts'] = fast_counts.fast_counts / fast_counts.all_counts

#selecting cases with 4+ e-mails sent/received and with 50%+ of fast responses
sel_fast_cases_idx = np.where((fast_counts.perc_fast_counts >= 0.5) & (fast_counts.all_counts >= 4))[0]
fast_counts = fast_counts.iloc[sel_fast_cases_idx,:].copy()
fast_counts.reset_index(drop = True, inplace = True)


#merge to prune out invalid clientIDs with sanity check
print(fast_counts.shape)
fast_counts = fast_counts.merge(data_viz2, how="left", left_on="clientID", right_on = 'clientID').copy().reset_index(drop = True)
print(len(fast_counts.clientID.unique()))
#done

#removing all non-fast cases w/ sanity check
print(len(fast_counts.clientID.unique()))
sel_cases_idx = np.where(fast_counts.email_proc_restime < min_th_fast_response)[0]
sel_cases_idx = fast_counts.index[sel_cases_idx]
fast_counts = fast_counts.iloc[sel_cases_idx,:].copy()
fast_counts.reset_index(drop = True,  inplace = True)
print(len(fast_counts.clientID.unique()))

#counting occurrences in different periods of the day
df = fast_counts.groupby('clientID')['email_proc_obsblock'].value_counts().unstack()
df = df.count(axis=1).to_frame()
df.reset_index(level=0, inplace=True)
df = df.rename(columns={0: "different_periods"})

#excluding cases which occur always in the same period of the day w/sanity check
print(df.shape)
sel_cases_idx = np.where(df.different_periods > 1)[0]
sel_cases_idx = df.index[sel_cases_idx]
df = df.iloc[sel_cases_idx,:].copy()
df.reset_index(drop = True,  inplace = True)
print(df.shape)

#adding info to original test set, w/ sanity check
print(df.shape)
test_data_melted['email_proc_fast_responder'] = 0
sel_cases_idx = np.where(test_data_melted.clientID.isin(df.clientID))[0]
sel_cases_idx = test_data_melted.index[sel_cases_idx]
test_data_melted.loc[sel_cases_idx,'email_proc_fast_responder'] = 1
print(len(test_data_melted.clientID[sel_cases_idx].unique()))

test_data_melted = test_data_melted.filter(['clientID', 'client_X1', 'client_X2', 'client_X3', 'email_proc_obstime', 'email_proc_obshour', 'email_proc_restime', 'email_proc_obsblock', 'email_proc_fast_responder','email_M'])

#
#compare distributions of new features between train and test
#
training_data_melted.drop(columns='clientID').hist(figsize=(30, 30))
test_data_melted.drop(columns='clientID').hist(figsize=(30, 30))
#the distributions seem similar, as expected.

#storing resulting objects
training_data_melted.to_csv(path + "proc.data/training_data.csv", index=False)
test_data_melted.to_csv(path + "proc.data/test_data.csv", index=False)



