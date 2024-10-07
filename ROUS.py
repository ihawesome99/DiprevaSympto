import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'over_undersampling.csv'
df = pd.read_csv(file_path)

df.head()
df = pd.read_csv('over_undersampling.csv')

# Set variabel target
print(df['assesment'].value_counts())

# Define sampling strategy
sampling_strategy = {label: 236 for label in df['assesment'].unique()}

# Create Pipeline untuuk Over dan Under
pipeline = Pipeline([
    ('over', RandomOverSampler(sampling_strategy='auto')),
    ('under', RandomUnderSampler(sampling_strategy=sampling_strategy))
])

# Apply pipeline untuk variabel label dan fitur
df_features = df.drop('assesment', axis=1)
df_target = df['assesment']
X_resampled, y_resampled = pipeline.fit_resample(df_features, df_target)

# Combine resampled data dan buat dataframe baru
df_resampled = pd.DataFrame(X_resampled, columns=df_features.columns)
df_resampled['assesment'] = y_resampled

# Export the balanced dataset to Excel
balanced_file_path = 'balanced_dataset.xlsx'
df_resampled.to_excel(balanced_file_path, index=False)

# Show the balance of the new dataset
df_resampled['assesment'].value_counts().plot(kind='bar')
plt.title('Balanced Dataset')
plt.xlabel('Assesment')
plt.ylabel('Count')
plt.show()

print('Balanced dataset exported to:', balanced_file_path)