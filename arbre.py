# %% [markdown]
# <a href="https://colab.research.google.com/github/andreaaraldo/machine-learning-for-networks/blob/master/05.trees-and-ensembles/05.trees-and-ensembles.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

from collections import Counter

from imblearn.over_sampling import SMOTE


# Import the visualization library I prepared for you
! wget https://raw.githubusercontent.com/andreaaraldo/machine-learning-for-networks/master/course_library/visualization.py
from visualization import plot_conf_mat, plot_feature_importances


!pip install ipython-autotime # To show time at each cell
                              # Credits to https://medium.com/@arptoth/how-to-measure-execution-time-in-google-colab-707cc9aad1c8
%load_ext autotime


# The following is to be able to mount Google Drive
from google.colab import drive

import pickle # To save the model
from os.path import isfile

# %% [markdown]
# Mount Google Drive

# %%
mount_point = '/content/gdrive' # Always the same, don't change it
drive.mount(mount_point, force_remount=True)
drive_path = mount_point + '/My Drive/' # Always the same, don't change it
my_path = drive_path + \
  'tsp/teaching/data-science-for-networks/img-from-code/05.trees/'

# %% [markdown]
# # Use case and dataset
# 
# We use the dataset by [Reyhane Askari Hemmat](https://github.com/ReyhaneAskari/SLA_violation_classification) (Université de Montréal) used in [He16]. This dataset is built from [Google Cloud Cluster Trace](https://github.com/google/cluster-data), a 29-days trace of activity in a Google Cloud cluster. The trace reports:
# 
# * Resources available on the machines
# * Tasks submitted by users, along with the requested resources
# * Actual resources used by tasks
# * Events, like eviction of tasks (for lack of resources, failure of the machine, etc.)
# 
# 
# Hemmat et Al. [He16] pre-processed this trace:
# * For each submitted task, they checked if the task correctly terminates or is evicted
# * They created as csv file with the task characteristics and a `violation` column, to indicating failure (1) or normal termination (0).
# 
# 
# ### Goal
# Predict a task failure, i.e., whether a task [will be evicted](https://github.com/ReyhaneAskari/SLA_violation_classification/blob/55bba2683dec43e739244b6b616294827a98f8e1/3_create_database/scripts/full_db_2.py#L33) before normal termination.

# %%
!wget https://raw.githubusercontent.com/ReyhaneAskari/SLA_violation_classification/master/3_create_database/csvs/frull_db_2.csv

# %% [markdown]
# Unfortunately, [no GPU support](https://stackoverflow.com/a/41568439/2110769) is available for scikit learn.

# %% [markdown]
# # Load dataset and preliminary operations
# 

# %%
train_path = "frull_db_2.csv"
df = pd.read_csv(train_path)
df

# %% [markdown]
# Column description:
# * `job_id`: users submit jobs, i.e., a set of tasks
# * `task_idx`: the index of a task within a job. A task is uniquely identified by `(job_id, task_idx)`
# * `sched_cls`: From [Re11]: "3 representing a more latency-sensitive task (e.g., serving revenue-generating user requests) and 0 representing a non-production task (e.g., development, non-business-critical analyses, etc.)... more latency-sensitive tasks tend to have higher task priorities"
# * `priority`
# * `cpu_requested`: Maximum amount of CPU the task is permitted to use.
#   * Unit of measurement: core-count / second.
#   * The scale is relateive to the CPU available in the most powerful machine of the cluster.
#   * This is specified by the user at submission time
# * `mem_requested`: Maximum amount of memory the task is permitted to use.
#   * Unit of measurement: GB
#   * The scale is relateive to the memory available in the machine of the cluster with the largest memory.
#   * This is specified by the user at submission time
# * `disk`: Similarly to `mem_requested`

# %% [markdown]
# We need to remove features that have no predictive meaning

# %%
df = df.drop(labels=['Unnamed: 0', 'job_id', 'task_idx'], axis=1)
df

# %% [markdown]
# Let's partition the dataset in training and test dataset

# %%
X = df.drop(labels='violation', axis=1)
y = df['violation']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,
                                        shuffle=True, random_state=4)

# %% [markdown]
# Check for class imbalance and correct for it

# %%
print( "Samples per class before SMOTE: ", Counter(y_train) )

smote = SMOTE()
X_train, y_train = smote.fit_sample(X_train, y_train)

print( "Samples per class after SMOTE: ", Counter(y_train) )

# %% [markdown]
# # Training and testing a random forest

# %%
model = RandomForestClassifier(n_estimators=100,
                      criterion='gini',
                      max_leaf_nodes=16, # Each tree cannot have more than that
                      random_state=5, # For reproducibility
                      n_jobs=-1, # Use all the CPUs
                      max_features = 'auto' # auto means=sqrt(n_features)
                      )

model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)

class_names = np.array(['ok', 'fail'])
plot_conf_mat(y_test, y_pred, class_names)

# %% [markdown]
# Let's check the feature importance

# %%
feature_names = X_test.columns
plot_feature_importances(model.feature_importances_, feature_names)
feature_names

# %% [markdown]
# Memory and Disk are the most determining factors in producing a failure

# %% [markdown]
# # Hyperparameter tuning
# You have three possibility:
# * Manual tuning:
#   * Divide the training set in training subset and validation subset
#   * Train different classifiers (with different hyperparameters) on the training subset
#   * Check their performance, i.e., accuracy, on the validation subset
#   * Choose the best
#   * Test it on the test set
# * `GridSearchCV` (as in `02.regression/b.polynomial-regression.ipynb`)
# * `RandomizedSearchCV`
# 
# We use the latter now. See [Open Data Science post](https://medium.com/@ODSC/optimizing-hyperparameters-for-random-forest-algorithms-in-scikit-learn-d60b7aa07ead).
# 
# Let's first define the values of the parameters we want to explore

# %%
 param_grid = {
    'criterion':['gini', 'entropy'],
    'max_features':[1,2,3,4,5],

    # Number of allowed leafs
    'max_leaf_nodes':[16, 32, 64, 128, 256, 512, 1024, 2048],

    # A node will be split if this split induces a decrease of the
    # impurity greater than or equal to this value.
    'min_impurity_decrease' : [0, 0.001, 0.01, 0.1, 0.2],

    'max_depth':[1,10,100,1000,10000,100000],

    # A node can be a leaf only if it contains at least the following fraction
    # of samples
    'min_weight_fraction_leaf' : [0.1, 0.01, 0.001, 0]

}

# %% [markdown]
# We have a lot of possible configurations to check. We specify to just test 50
# out of them.
# 

# %%
# Before we had used all the availble CPUs for training one random forest.
# Now, instead, we use one CPU per random forest (n_jobs=1).
forest = RandomForestClassifier(n_estimators=100, random_state = 4, n_jobs=1,)


search = RandomizedSearchCV(
                            scoring = 'accuracy', # See other possible metrics in
                                                # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

                            estimator=forest,
                            param_distributions=param_grid,
                            n_iter=50, # We just test 50 configurations
                            verbose=2,random_state=42,
                            n_jobs=-1, # Triain forests in parallel using
                                      # all CPUs
                            cv=5 # 5-fold validation
                          )
# Note that we are training different random forests in parallel (n_jobs=-1),
# each with a certain combination of hyper-parameters.

search.fit(X_train, y_train)

# %%
print(search.best_params_)

model = search.best_estimator_


# %% [markdown]
# Now that we have the model with the best hyperparameters, we train it on the entire dataset

# %%
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plot_conf_mat(y_test, y_pred, class_names)

# %% [markdown]
# Note that we get also get probabilities with RandomForests (fraction of trees that predicted `1`). And thus, we can compute cross-entropy

# %%
proba = model.predict_proba(X_test)
print("Output probabilities are ",proba)


print("Cross entropy = ", log_loss(y_test, proba))

# %% [markdown]
# Don't confuse cross-entropy with entropy of a tree node!

# %% [markdown]
# Since it may take some time to perform randomized search, let's save the model (**serializing** the python object)

# %%
filename = my_path+'forest-1.pkl'
pickle.dump(model, open(filename, 'wb'))

# %% [markdown]
# To later retrieve it:
# 
# 

# %%
model = pickle.load(open(filename, 'rb'))

# %% [markdown]
# It is convenient to automate this process

# %%
def search_or_load(model_filename, search, X_train, y_train):
  if(isfile(model_filename) ):
    print("Loading model")
    model = pickle.load(open(model_filename, 'rb'))

  else:
    print("Searching the best hyper_parameters")
    search.fit(X_train, y_train)
    print(search.best_params_)
    model = search.best_estimator_
    print("Training model")
    model.fit(X_train,y_train)
    pickle.dump(model, open(model_filename, 'wb'))
    print("Model saved in in file ", model_filename)
  return model


# %% [markdown]
# If we call this function, it will not redo the search, as the model has already been saved

# %%
model = search_or_load(filename, search, X_train, y_train)

# %% [markdown]
# ### More iterations
# 
# 
# Let's try to increase the number of tested configurations
# 

# %%
search = RandomizedSearchCV(
                            scoring = 'accuracy', # See other possible metrics in
                                                # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

                            estimator=forest,
                            param_distributions=param_grid,
                            n_iter=200,
                            verbose=2,random_state=42,
                            n_jobs=-1, # Triain forests in parallel using
                                      # all CPUs
                            cv=5 # 5-fold validation
                          )



# %%
filename = my_path + 'forest-2.pkl'
model = search_or_load(filename, search, X_train, y_train)

# %%

y_pred = model.predict(X_test)

plot_conf_mat(y_test, y_pred, class_names)

# %% [markdown]
# It's better
# 
# Ways to improve eve further:
# * Increase the number of trees
# * Increase the number of configurations to try out

# %% [markdown]
# # Random forest for regression
# 
# We have performed in this notebook a classification task. If you need to perform a regression task instead, you can use the [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

# %% [markdown]
# # References
# 
# [He16] Hemmat, R. A., & Hafid, A. (2016). SLA Violation Prediction In Cloud Computing: A Machine Learning Perspective. Retrieved from http://arxiv.org/abs/1611.10338
# 
# [Re11] Reiss, C., Wilkes, J., & Hellerstein, J. (2011). Google cluster-usage traces: format+ schema. Google Inc., …, 1–14. https://doi.org/10.1007/978-3-540-69057-3_88

# %%



