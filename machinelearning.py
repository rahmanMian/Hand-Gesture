import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from flask import Flask, render_template

# Function to load the data from JSON files
def load_data(direction, num_files):
    dfs = []
    for i in range(num_files):
        file_name = f'{direction}_{i}.log'

        with open(file_name, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df['time'] = df['time'].str.replace('ms', '').astype(float)
        dfs.append(df)
    return pd.concat(dfs)

# Load the data from all the JSON files
left_df = load_data('left', 9)
up_df = load_data('up', 9)
right_df = load_data('right', 9)
down_df = load_data('down', 9)

# Add a direction column to each DataFrame
left_df['direction'] = 1
up_df['direction'] = 0
right_df['direction'] = 0
down_df['direction'] = 1

# Combine the data into one DataFrame
df = pd.concat([left_df, up_df, right_df, down_df])

# Extract features and target variable
X = df[['acce_x', 'acce_y', 'acce_z', 'gyro_x', 'gyro_y', 'gyro_z']]
y = df['direction']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model's performance
y_pred = clf.predict(X_test)
print(y_pred)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred, average='weighted'))
print('Recall:', recall_score(y_test, y_pred, average='weighted'))
print('F1 Score:', f1_score(y_test, y_pred, average='weighted'))


def predict_direction_from_json(json_file_path, clf):
    # Read the entire file as a string
    with open(json_file_path) as f:
        file_contents = f.read()

    # Find the start and end indices of the JSON array
    start_index = file_contents.find('[')
    end_index = file_contents.rfind(']') + 1  # Add 1 to include the closing bracket

    # Extract the JSON array substring
    json_array_str = file_contents[start_index:end_index]

    # Load the JSON array
    data = json.loads(json_array_str)

    # Convert the 'time' column to a timedelta object
    df = pd.DataFrame(data)
    df['time'] = pd.to_timedelta(df['time'])

    # Extract the features and predict the direction using the trained model
    X = df[['acce_x', 'acce_y', 'acce_z', 'gyro_x', 'gyro_y', 'gyro_z']]
    y_pred = clf.predict(X)

    # Get the index of the maximum value in the predicted label array
    direction_index = np.argmax(y_pred)

    # Map the index to the corresponding direction string
    if direction_index == 0:
        predicted_direction = 'MOVEMENT IN HORIZONTAL PLANE'
    elif direction_index == 1:
        predicted_direction = 'MOVEMENT IN VERTICAL PLANE'

    return predicted_direction



app = Flask(__name__)


# Define a route to render the form
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle the form submission
@app.route('/predict', methods=['GET'])
def predict():
    # Get the JSON file path from the query parameters
  # Process the JSON file path and return it as plain text
    print("yo")
    return predict_direction_from_json("testing.json", clf)

if __name__ == '__main__':
      app.run() 