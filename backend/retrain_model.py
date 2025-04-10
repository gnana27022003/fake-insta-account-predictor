import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# ===== 1. Load the Data =====
train = pd.read_csv(r'C:\Users\User\Desktop\FakePredict\backend\data\train.csv')
test = pd.read_csv(r'C:\Users\User\Desktop\FakePredict\backend\data\test.csv')

# Split into features and target
X_train = train.drop('fake', axis=1)
y_train = train['fake']

X_test = test.drop('fake', axis=1)
y_test = test['fake']

# ===== 2. Apply SMOTE to Balance the Training Set =====
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\nClass Distribution After SMOTE:")
print(pd.Series(y_train_balanced).value_counts())

# ===== 3. Reduce `profile pic` Influence =====
# Identify the index of 'profile pic'
profile_pic_index = list(X_train.columns).index('profile pic')

# Apply scaling factor (reduce influence by 90%)
scaling_factor = 0.1
X_train_balanced.iloc[:, profile_pic_index] *= scaling_factor
X_test.iloc[:, profile_pic_index] *= scaling_factor

# ===== 4. Scale the Data =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# ===== 5. Hyperparameter Tuning (for RandomForest) =====
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use GridSearchCV to find the best model
print("\nRunning Hyperparameter Tuning...")
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train_scaled, y_train_balanced)

# Use the best model
best_model = grid_search.best_estimator_
print(f"\nBest Parameters: {grid_search.best_params_}")

# ===== 6. Train the Model (Switch Between XGBoost and RandomForest) =====
use_xgboost = False  # Set to True for XGBoost, False for RandomForest

if use_xgboost:
    print("\nTraining XGBoost Model...")
    model = XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
    model.fit(X_train_scaled, y_train_balanced)
else:
    print("\nTraining RandomForest Model...")
    model = best_model

# ===== 7. Evaluate the Model =====
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# ===== 8. Visualize Feature Importance =====
if hasattr(model, "feature_importances_"):
    feature_importances = model.feature_importances_
    feature_names = X_train.columns

    plt.figure(figsize=(14, 7))
    plt.barh(feature_names, feature_importances, color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance After Reducing `profile pic` Influence')
    plt.show()

# ===== 9. Save the Model and Scaler =====
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("\nModel and scaler saved successfully!")
