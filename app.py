from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the dataset and preprocess
df = pd.read_csv('covtype.csv')
df_without_soil = df.drop(df.columns[10:54], axis=1)
features = df_without_soil.drop('Cover_Type', axis=1)
target = df_without_soil['Cover_Type']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the Random Forest model
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    
    # Rename the input data keys to match the feature names used during training
    input_data = {
        'Elevation': float(data['elevation']),
        'Aspect': float(data['aspect']),
        'Slope': float(data['slope']),
        'Horizontal_Distance_To_Hydrology': float(data['horizontal_hydro']),
        'Vertical_Distance_To_Hydrology': float(data['vertical_hydro']),
        'Horizontal_Distance_To_Roadways': float(data['horizontal_road']),
        'Hillshade_9am': float(data['hillshade_9am']),
        'Hillshade_Noon': float(data['hillshade_noon']),
        'Hillshade_3pm': float(data['hillshade_3pm']),
        'Horizontal_Distance_To_Fire_Points': float(data['horizontal_fire'])
    }
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame(input_data, index=[0])
    
    # Make prediction
    prediction = random_forest.predict(input_df)
    
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
