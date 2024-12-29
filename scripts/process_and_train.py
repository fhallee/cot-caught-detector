import argparse
import numpy as np
import os
import pandas as pd
import parselmouth
import textgrids
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Set up argument parser
parser = argparse.ArgumentParser(description='Process formant data and train a Random Forest classifier.')
parser.add_argument('--textgrid_dir', type=str, required=True, help='Path to the directory containing TextGrid files')
parser.add_argument('--sound_dir', type=str, required=True, help='Path to the directory containing sound files')
args = parser.parse_args()

# Directory Paths
textgrid_dir = args.textgrid_dir
sound_dir = args.sound_dir

# Create an empty DataFrame to store the formant data
formant_data = pd.DataFrame(columns=['Dialect Region', 'Gender', 'Speaker ID', 'Sentence #', 'Vowel', 'f1', 'f2', 'f3', 'f4', 'vowel_length'])

# Read data into DataFrame
for filename in os.listdir(textgrid_dir):
    filepath = os.path.join(textgrid_dir, filename)
    # Splits the filename into dialect region, gender, speaker id, and sentence number
    if len(filename.split("_")) == 3:
        dialect_region, gender_speaker_id, sentence_num = filename.split("_")
        gender, speaker_id = gender_speaker_id[0], gender_speaker_id[1:]
        sentence_num = sentence_num.split(".")[0]
        if dialect_region not in ['DR6', 'DR7']:
            continue
        try:
            # Load the textgrid
            grid = textgrids.TextGrid(filepath)
            # Load the corresponding sound file
            sound = parselmouth.Sound(os.path.join(sound_dir, filename.replace("TextGrid", "wav")))
            # Extract formants from the sound
            formants = sound.to_formant_burg(maximum_formant=5000)
            # Iterate through the phonemes
            for phoneme in grid['phones']:
                if phoneme.containsvowel() and phoneme.text != "sil":
                    vowel = phoneme.text.transcode(retain_diacritics=True)
                    # Calculate the midpoint of the vowel
                    midpoint = (phoneme.xmin + phoneme.xmax) / 2
                    # Calculate the vowel length
                    vowel_length = phoneme.xmax - phoneme.xmin
                    # Extract the first four formants at the midpoint
                    f1 = formants.get_value_at_time(1, midpoint)
                    f2 = formants.get_value_at_time(2, midpoint)
                    f3 = formants.get_value_at_time(3, midpoint)
                    f4 = formants.get_value_at_time(4, midpoint)
                    # Check for NaN values
                    if np.isnan(f1) or np.isnan(f2) or np.isnan(f3) or np.isnan(f4):
                        print(f"Skipping phoneme with NaN values in file {filename}")
                        continue
                    new_row = {'Dialect Region': dialect_region, 'Gender': gender, 'Speaker ID': speaker_id,
                                'Sentence #': sentence_num, 'Vowel': vowel, 'f1': round(f1), 'f2': round(f2), 'f3': round(f3), 'f4': round(f4), 'vowel_length': vowel_length}
                    formant_data = pd.concat([formant_data, pd.DataFrame([new_row])], ignore_index=True)
        except FileNotFoundError as e:
            print(f"Error processing {filename}: {e}")
        except Exception as e:
            print(f"Unexpected error processing {filename}: {e}")
    else:
        print(f"Invalid filename format: {filename}")

# Rename dialect regions
formant_data['Dialect Region'] = formant_data['Dialect Region'].replace({'DR7': 'Merger', 'DR6': 'No Merger'})

# Initialize dictionary to store mean and std deviation for each speaker
speaker_stats = {}

# Iterate through each speaker and calculate mean and std deviation for each formant
for speaker_id, group in formant_data.groupby('Speaker ID'):
    f1_mean = float(group['f1'].mean())
    f1_std = float(group['f1'].std())
    f2_mean = float(group['f2'].mean())
    f2_std = float(group['f2'].std())
    f3_mean = float(group['f3'].mean())
    f3_std = float(group['f3'].std())
    f4_mean = float(group['f4'].mean())
    f4_std = float(group['f4'].std())
    vowel_length_mean = float(group['vowel_length'].mean())
    vowel_length_std = float(group['vowel_length'].std())
    
    speaker_stats[speaker_id] = {
        'f1_mean': f1_mean,
        'f1_std': f1_std,
        'f2_mean': f2_mean,
        'f2_std': f2_std,
        'f3_mean': f3_mean,
        'f3_std': f3_std,
        'f4_mean': f4_mean,
        'f4_std': f4_std,
        'vowel_length_mean': vowel_length_mean,
        'vowel_length_std': vowel_length_std
    }

# Filter the data to include only the vowels 'aa' and 'ao'
formant_data = formant_data[formant_data['Vowel'].isin(['aa', 'ao'])]

# Apply Lobanov normalization
def lobanov_normalize(row):
    speaker_id = row['Speaker ID']
    stats = speaker_stats[speaker_id]
    row['f1'] = (row['f1'] - stats['f1_mean']) / stats['f1_std']
    row['f2'] = (row['f2'] - stats['f2_mean']) / stats['f2_std']
    row['f3'] = (row['f3'] - stats['f3_mean']) / stats['f3_std']
    row['f4'] = (row['f4'] - stats['f4_mean']) / stats['f4_std']
    row['vowel_length'] = (row['vowel_length'] - stats['vowel_length_mean']) / stats['vowel_length_std']
    return row

formant_data = formant_data.apply(lobanov_normalize, axis=1)

# Initialize dictionary to store final data
final_data = {}

# Iterate through each speaker and calculate average for each normalized formant, separated by vowels
for speaker_id, group in formant_data.groupby('Speaker ID'):
    aa_group = group[group['Vowel'] == 'aa']
    ao_group = group[group['Vowel'] == 'ao']
    
    final_data[speaker_id] = {
        'aa': {
            'f1_mean': aa_group['f1'].mean(),
            'f2_mean': aa_group['f2'].mean(),
            'f3_mean': aa_group['f3'].mean(),
            'f4_mean': aa_group['f4'].mean(),
            'vowel_length_mean': aa_group['vowel_length'].mean()
        },
        'ao': {
            'f1_mean': ao_group['f1'].mean(),
            'f2_mean': ao_group['f2'].mean(),
            'f3_mean': ao_group['f3'].mean(),
            'f4_mean': ao_group['f4'].mean(),
            'vowel_length_mean': ao_group['vowel_length'].mean()
        }
    }

# Convert final_data to DataFrame for further processing
final_df = pd.DataFrame.from_dict(final_data, orient='index')
final_df = final_df.reset_index().rename(columns={'index': 'Speaker ID'})

# Flatten the DataFrame
final_df = pd.concat([final_df.drop(['aa', 'ao'], axis=1), final_df['aa'].apply(pd.Series).add_suffix('_aa'), final_df['ao'].apply(pd.Series).add_suffix('_ao')], axis=1)

# Add the dialect region back to the final_df
final_df = final_df.merge(formant_data[['Speaker ID', 'Dialect Region']].drop_duplicates(), on='Speaker ID')

# Print rows with missing values
print("Rows with missing values:")
print(final_df[final_df.isnull().any(axis=1)])

# Drop rows with missing values
final_df = final_df.dropna()

# Train-test split
X = final_df.drop(['Speaker ID', 'Dialect Region'], axis=1)
y = final_df['Dialect Region']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the parameter grid for Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, refit=True, verbose=2, cv=5)
grid_search.fit(X_train, y_train)

# Get the best estimator
best_rf = grid_search.best_estimator_

# Predict on the test set using the best Random Forest
y_pred_rf = best_rf.predict(X_test)

# Evaluate the best Random Forest classifier
accuracy_rf = accuracy_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf, pos_label='Merger')
precision_rf = precision_score(y_test, y_pred_rf, pos_label='Merger')
f1_rf = f1_score(y_test, y_pred_rf, pos_label='Merger')

print("Best Hyperparameters:")
print(grid_search.best_params_)

print("Best Random Forest:")
print(f"Accuracy: {accuracy_rf}")
print(f"Recall: {recall_rf}")
print(f"Precision: {precision_rf}")
print(f"F1 Score: {f1_rf}")

# Feature importances
importances = best_rf.feature_importances_
feature_names = X_train.columns

# Create a DataFrame for better visualization
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(feature_importances)