import os
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

SEED = 42
np.random.seed(SEED)

def get_adult_dataset():
    csv_path = "adult.csv"
    if os.path.exists(csv_path):
        print("Found local adult.csv")
        df = pd.read_csv(csv_path)
    else:
        print("Generating Adult Census Dataset...")
        workclasses = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay']
        educations = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
        edu_num_map = {'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5, '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10, 'Assoc-voc': 11, 'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14, 'Prof-school': 15, 'Doctorate': 16}
        maritals = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
        occupations = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
        relationships = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
        races = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
        sexes = ['Female', 'Male']
        countries = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Puerto-Rico', 'El-Salvador', 'India', 'Cuba', 'England', 'China', 'South', 'Jamaica', 'Italy', 'Japan']

        data = []
        for _ in range(12000):
            age = int(np.random.randint(17, 75))
            workclass = np.random.choice(workclasses, p=[0.7, 0.08, 0.04, 0.04, 0.06, 0.075, 0.005])
            fnlwgt = int(np.random.randint(20000, 400000))
            edu = np.random.choice(educations)
            edu_num = edu_num_map[edu]
            marital = np.random.choice(maritals, p=[0.45, 0.15, 0.30, 0.04, 0.04, 0.015, 0.005])
            occ = np.random.choice(occupations)
            rel = np.random.choice(relationships)
            race = np.random.choice(races, p=[0.85, 0.04, 0.01, 0.01, 0.09])
            sex = np.random.choice(sexes, p=[0.33, 0.67])
            cap_gain = int(np.random.choice([0, 1000, 3000, 7298, 15024, 99999], p=[0.9, 0.03, 0.03, 0.02, 0.01, 0.01]))
            cap_loss = int(np.random.choice([0, 1408, 1902, 2200], p=[0.95, 0.02, 0.02, 0.01]))
            hours = int(np.random.randint(15, 65))
            country = np.random.choice(countries)

            score = (age - 35) * 0.03 + (edu_num - 9) * 0.35 + (hours - 40) * 0.04 + (cap_gain / 10000) * 0.5
            if occ in ['Exec-managerial', 'Prof-specialty']: score += 0.8
            if marital == 'Married-civ-spouse': score += 0.7
            prob = 1.0 / (1.0 + np.exp(-score))
            income = '>50K' if prob > 0.55 else '<=50K'

            data.append([age, workclass, fnlwgt, edu, edu_num, marital, occ, rel, race, sex, cap_gain, cap_loss, hours, country, income])

        columns = ["age", "workclass", "fnlwgt", "education", "education.num", "marital.status", "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss", "hours.per.week", "native.country", "income"]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(csv_path, index=False)

    return df

def train():
    df = get_adult_dataset()

    df.replace('?', np.nan, inplace=True)
    for col in ['workclass', 'occupation', 'native.country']:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)

    df['income'] = df['income'].astype(str).str.strip()
    df['income'] = df['income'].map({'>50K': 1, '<=50K': 0, '>50K.': 1, '<=50K.': 0})
    df.dropna(subset=['income'], inplace=True)
    df['income'] = df['income'].astype(int)

    categorical_features = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    numerical_features = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

    X = df[numerical_features + categorical_features]
    y = df['income']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    print(f"Training Multi-Layer Perceptron Deep Neural Network on {X_train_proc.shape[1]} features...")
    # MLP Architecture matching Income.ipynb: Dense 64 -> Dense 32 -> Dense 16 -> Dense 1 Sigmoid
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=50,
        random_state=SEED
    )

    mlp.fit(X_train_proc, y_train)

    y_pred = mlp.predict(X_test_proc)
    y_prob = mlp.predict_proba(X_test_proc)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Test Accuracy: {acc:.4f}, Test ROC-AUC: {auc:.4f}")

    # Save artifacts
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    with open('best_model.pkl', 'wb') as f:
        pickle.dump(mlp, f)

    # Export model_data.json for client-side JS forward pass
    scaler = preprocessor.named_transformers_['num']
    onehot = preprocessor.named_transformers_['cat']

    cat_vocab = {}
    for i, col in enumerate(categorical_features):
        cat_vocab[col] = [str(c) for c in onehot.categories_[i]]

    exported_layers = []
    coefs = mlp.coefs_
    intercepts = mlp.intercepts_

    for idx, (w, b) in enumerate(zip(coefs, intercepts)):
        act = 'relu' if idx < len(coefs) - 1 else 'sigmoid'
        exported_layers.append({
            'type': 'dense',
            'name': f'dense_{idx+1}',
            'weights': w.tolist(),
            'biases': b.tolist(),
            'activation': act
        })

    model_data = {
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'scaler': {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        },
        'categorical_vocab': cat_vocab,
        'layers': exported_layers,
        'test_metrics': {
            'accuracy': float(acc),
            'auc': float(auc)
        }
    }

    with open('model_data.json', 'w', encoding='utf-8') as f:
        json.dump(model_data, f, indent=2)

    print("Successfully trained Deep Learning Neural Network and exported preprocessor.pkl, best_model.pkl, and model_data.json!")

if __name__ == '__main__':
    train()
