import os
import json
import urllib.request
import numpy as np
import pandas as pd
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import class_weight

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def get_adult_dataset():
    csv_path = "adult.csv"
    if os.path.exists(csv_path):
        print("Found local adult.csv")
        df = pd.read_csv(csv_path)
    else:
        print("Downloading adult dataset...")
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"
        columns = [
            "age", "workclass", "fnlwgt", "education", "education.num",
            "marital.status", "occupation", "relationship", "race", "sex",
            "capital.gain", "capital.loss", "hours.per.week", "native.country", "income"
        ]
        try:
            df = pd.read_csv(url, header=None, names=columns)
            df.to_csv(csv_path, index=False)
            print("Successfully downloaded adult.csv")
        except Exception as e:
            print(f"Failed download: {e}, generating synthetic census data...")
            df = generate_synthetic_census_data()
            df.to_csv(csv_path, index=False)
    return df

def generate_synthetic_census_data(num_samples=10000):
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
    for _ in range(num_samples):
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

        score = (age - 35) * 0.03 + (edu_num - 9) * 0.3 + (hours - 40) * 0.04 + (cap_gain / 10000) * 0.5
        if occ in ['Exec-managerial', 'Prof-specialty']:
            score += 0.8
        if marital == 'Married-civ-spouse':
            score += 0.7
        prob = 1.0 / (1.0 + np.exp(-score))
        income = '>50K' if prob > 0.55 else '<=50K'

        data.append([age, workclass, fnlwgt, edu, edu_num, marital, occ, rel, race, sex, cap_gain, cap_loss, hours, country, income])

    columns = ["age", "workclass", "fnlwgt", "education", "education.num", "marital.status", "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss", "hours.per.week", "native.country", "income"]
    return pd.DataFrame(data, columns=columns)

def train_and_export():
    df = get_adult_dataset()

    # Data Cleaning matching Income.ipynb
    df.replace('?', np.nan, inplace=True)
    for col in ['workclass', 'occupation', 'native.country']:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)

    # Encode target
    df['income'] = df['income'].astype(str).str.strip()
    df['income'] = df['income'].map({'>50K': 1, '<=50K': 0, '>50K.': 1, '<=50K.': 0})
    df.dropna(subset=['income'], inplace=True)
    df['income'] = df['income'].astype(int)

    categorical_features = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    numerical_features = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

    X = df[numerical_features + categorical_features]
    y = df['income']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

    # ColumnTransformer matching Income.ipynb
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    input_dim = X_train_proc.shape[1]
    print(f"Processed feature count: {input_dim}")

    # Compute class weights
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = {0: float(weights[0]), 1: float(weights[1])}

    # Build Keras Model matching Income.ipynb architecture
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu', name='dense_1'),
        layers.BatchNormalization(name='bn_1'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', name='dense_2'),
        layers.BatchNormalization(name='bn_2'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu', name='dense_3'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid', name='dense_out')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
    ]

    print("Training Keras neural network model...")
    history = model.fit(
        X_train_proc, y_train,
        validation_data=(X_val_proc, y_val),
        epochs=35,
        batch_size=32,
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    eval_results = model.evaluate(X_test_proc, y_test, verbose=0)
    print(f"Test Accuracy: {eval_results[1]:.4f}, Test AUC: {eval_results[2]:.4f}")

    # Save preprocessor pickle
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    # Save model metadata and exported parameters for JS
    scaler = preprocessor.named_transformers_['num']
    onehot = preprocessor.named_transformers_['cat']

    cat_feature_values = {}
    for i, col in enumerate(categorical_features):
        cat_feature_values[col] = [str(c) for c in onehot.categories_[i]]

    # Extract model weights for client-side JS forward pass
    exported_layers = []
    for layer in model.layers:
        l_name = layer.name
        weights = layer.get_weights()
        if 'dense' in l_name and len(weights) == 2:
            w, b = weights
            exported_layers.append({
                'type': 'dense',
                'name': l_name,
                'weights': w.tolist(),
                'biases': b.tolist(),
                'activation': layer.activation.__name__
            })
        elif 'bn' in l_name and len(weights) == 4:
            gamma, beta, m_mean, m_var = weights
            eps = float(layer.epsilon)
            exported_layers.append({
                'type': 'batch_norm',
                'name': l_name,
                'gamma': gamma.tolist(),
                'beta': beta.tolist(),
                'moving_mean': m_mean.tolist(),
                'moving_variance': m_var.tolist(),
                'epsilon': eps
            })

    model_data = {
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'scaler': {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        },
        'categorical_vocab': cat_feature_values,
        'layers': exported_layers,
        'test_metrics': {
            'accuracy': float(eval_results[1]),
            'auc': float(eval_results[2])
        }
    }

    with open('model_data.json', 'w', encoding='utf-8') as f:
        json.dump(model_data, f, indent=2)

    print("Model training, preprocessor, and model_data.json exported successfully!")

if __name__ == '__main__':
    train_and_export()
