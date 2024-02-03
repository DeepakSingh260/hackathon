import pandas 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

patient_data = pandas.read_csv(
    "Disease_symptom_and_patient_profile_dataset.csv",
    
)

X = patient_data[['Disease','Fever','Cough','Fatigue','Difficulty Breathing','Age','Gender','Blood Pressure','Cholesterol Level']]
y = patient_data['Outcome Variable']

