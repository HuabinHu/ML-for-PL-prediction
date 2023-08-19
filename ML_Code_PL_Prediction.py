import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    matthews_corrcoef, confusion_matrix, roc_auc_score,
    roc_curve, balanced_accuracy_score, f1_score, make_scorer,accuracy_score,recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

# Functions
def get_dataframe(path):
    """Read a CSV file and return a DataFrame."""
    return pd.read_csv(path, sep="\t")

def fp_as_array(mol, fp_fn):
    """Calculate a molecular fingerprint and convert it into a numpy array."""
    fp = fp_fn(mol)
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def get_MACCS_fp(mol):
    """Calculate the MACCS fingerprint of a molecule."""
    return MACCSkeys.GenMACCSKeys(mol)

def label_results(truth_list, pred_list):
    """Function to classify the prediction results."""
    label_list = [["TN","FN"],["FP","TP"]]
    res = [] 
    for i, j in zip(pred_list, truth_list):
        res.append(label_list[i][j])

# Load data,change location to your data dir
df = get_dataframe("../Curated data set.csv")

# Add fingerprints to DataFrame
sf = "SMILE" #Field with the SMILES
df['Mol'] = [Chem.MolFromSmiles(x) for x in df[sf]]
fp_list = [fp_as_array(x, get_MACCS_fp) for x in df.Mol]
df["fp"] = fp_list

# Add additional features to fingerprint
df["fp_MACCS"] = [np.append(df.loc[i, "fp"], df.loc[i, "CAT_pka"]) for i in range(df.shape[0])]
df["fp_full"] = [np.append(df.loc[i, "fp_MACCS"], df.loc[i, "CAT_logP"]) for i in range(df.shape[0])]


# Define model parameters
data_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=56)
model_param_dict = {'param_grid':{'n_estimators': [100, 200, 300, 400], 
                                  'bootstrap': [False],
                                  'max_features': ["sqrt", "log2", 0.7], 
                                  'max_depth': [7, 10, 12, None],
                                  'class_weight': ['balanced'],
                                  'min_samples_leaf': [1, 3, 5, 10]},
                   'cv': data_splitter,
                   'scoring': ["balanced_accuracy"],
                   'refit': "balanced_accuracy",
                   'n_jobs': -1}

stat_list = []
detail_list = []
pred_list = []
prob_list = {}
result_list = []

# Run the training and evaluation process for ten cycles with different random states
for cycle in tqdm(range(0,10)):
    # Split the dataset into training and testing sets
    train, test = train_test_split(df, test_size=0.3, random_state=cycle)
    
    # Extract the features and labels from the training and testing sets
    train_x, test_x, train_y, test_y = list(train.fp_full), list(test.fp_full), train.Class, test.Class 
    
    # Perform a grid search to find the best hyperparameters for a random forest classifier
    rf = GridSearchCV(RandomForestClassifier(random_state=cycle), **model_param_dict,verbose=10)
    rf.fit(train_x, train_y)
    
    # Make predictions on the test set
    pred = rf.predict(test_x) 
    pred_list.append([pred, test_y])
    result_list = label_results(test_y,pred)
    
    # Calculate the performance statistics for the predictions
    accuracy = accuracy_score(test_y, pred)
    recall = recall_score(test_y, pred)
    fi_score = f1_score(test_y, pred)
    stat_list.append({"accuracy": accuracy, "recall": recall, "fi_score": fi_score})
    
    # Calculate the predicted probabilities and prediction results for each test molecule
    prob = rf.predict_proba(test_x)
    prob_list[cycle] = [mol_prob[1] for mol_prob in prob]

# Print the average performance statistics across all cycles
print("Average Performance Statistics:")
print("accuracy: {:.4f}".format(sum(stat["accuracy"] for stat in stat_list) / len(stat_list)))
print("recall: {:.4f}".format(sum(stat["recall"] for stat in stat_list) / len(stat_list)))
print("F1 Score: {:.4f}".format(sum(stat["fi_score"] for stat in stat_list) / len(stat_list)))

# The ideal trail for prediction is achieved by setting the random state to 8;

#Showing the performance metrics in different trials
row_list = []
for p,t in pred_list:
    row_list.append(confusion_matrix(p,t).flatten())
confusion_df = pd.DataFrame(row_list,columns="tn,fn,fp,tp".split(","))
confusion_df["cycle"]=range(0,10)
confusion_df["accuracy"] = [x['accuracy'] for x in stat_list]
confusion_df["recall"] = [x['recall'] for x in stat_list]
confusion_df["F1 score"] = [x['fi_score'] for x in stat_list]
confusion_df
