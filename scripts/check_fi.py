import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from data_prep import load_data, prepare_data_before_split, encode_after_split
from config import DATA_FILE, TARGET_COLUMN

df = load_data(DATA_FILE)
df = prepare_data_before_split(df)

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN].astype(int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_test = encode_after_split(X_train, X_test, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("Top 20 Features:")
print(imp.head(20))
print("\nBottom 10 Features:")
print(imp.tail(10))
