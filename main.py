import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Optional but helpful
import kagglehub

# Download latest version
path = kagglehub.dataset_download("kamilpytlak/personal-key-indicators-of-heart-disease")

print("Path to dataset files:", path)

path = '2022/heart_2022_with_nans.csv'
data = pd.read_csv(path)
#print the data out
print(data.head())

# Step 1: List all columns and number of records
print("Columns:", data.columns)
print("Number of records:", data.shape[0])

# Step 2: Check for missing values
missing_values = data[data.isna()]
print("Missing values: ", missing_values)

# Step 3: Drop rows with missing values
data_cleaned = data.dropna()
print("Number of records after dropping missing values:", data_cleaned.shape[0])
print("Missing values after dropping: ", data_cleaned[data_cleaned.isna()])

#Step 4: Initial Visualizations

# Bar plots for categorical variables
categorical_cols = ['Sex', 'GeneralHealth']

for col in categorical_cols:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data_cleaned, x=col, hue='HadHeartAttack')
    plt.title(f'{col} Distribution by Heart Attack Status')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.legend(title='Had Heart Attack')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

#BMI and SleepHours are numeric columns
numeric_cols = ['BMI', 'SleepHours']

for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=data_cleaned, x=col, hue='HadHeartAttack', kde=True, bins=30)
    plt.title(f'{col} Distribution by Heart Attack Status')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
binary_cols = ['CovidPos', 'FluVaxLast12']

for col in binary_cols:
    # Cross-tab for better labeling
    cross = pd.crosstab(data_cleaned[col], data['HadHeartAttack'])

    for val in cross.index:
        plt.figure(figsize=(5, 5))
        plt.pie(cross.loc[val], labels=['No Heart Attack', 'Heart Attack'], 
                autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
        plt.title(f'Heart Attack Rate for {col} = {val}')
        plt.tight_layout()
        plt.show()


# Step 5: Correlation Matrix

# 1. Copy & encode your target as numeric
df_hm = data_cleaned.copy()
df_hm['HadHeartAttack_flag'] = df_hm['HadHeartAttack'].map({'No': 0, 'Yes': 1})

# 2. Identify your original numeric features (exclude the flag for now)
numeric_feats = df_hm.select_dtypes(include='number').columns.drop('HadHeartAttack_flag')

# 3. Build a full corr matrix including the flag
all_cols = list(numeric_feats) + ['HadHeartAttack_flag']
corr = df_hm[all_cols].corr()

# 4. Compute each feature’s correlation to the target
corr_with_target = corr['HadHeartAttack_flag']

# 5. Overwrite the diagonal entries for the original features
for feat in numeric_feats:
    corr.at[feat, feat] = corr_with_target[feat]

# 6. Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    annot=True,            # show correlation coefficients
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=.5
)
plt.title("Correlation Heatmap\n(diagonal = corr(feature, HadHeartAttack))")
plt.tight_layout()
plt.show()