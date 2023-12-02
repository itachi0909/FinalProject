from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('cars.csv')

# Select relevant features for recommendation
selected_features = ['Fuel', 'Seats', 'NCAP Rating', 'Fuel Efficiency']

# Filter the dataframe with selected features
df_selected = df[selected_features]

# Convert categorical variables to numerical using Label Encoding
label_encoders = {}
for column in df_selected.select_dtypes(include='object').columns:
    label_encoders[column] = LabelEncoder()
    df_selected[column] = label_encoders[column].fit_transform(df_selected[column])

# Handling missing values
df_selected['Fuel Efficiency'].fillna(df_selected['Fuel Efficiency'].mean(), inplace=True)

# Train the regression model
X = df_selected.drop('Fuel Efficiency', axis=1)  # Features
y = df_selected['Fuel Efficiency']  # Target variable
regression_model = RandomForestRegressor(n_estimators=100, random_state=42)
regression_model.fit(X, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        fuel_input = int(request.form['fuel'])
        seats_input = int(request.form['seats'])
        ncap_input = int(request.form['ncap'])

        # Prepare user input to match the feature format
        user_input = [[fuel_input, seats_input, ncap_input]]

        user_input_encoded = []
        for i, column in enumerate(df_selected.columns):
            if column != 'Fuel Efficiency':
                if df_selected[column].dtype == 'object':
                    user_input_encoded.append(label_encoders[column].transform([user_input[0][i]])[0])
                else:
                    user_input_encoded.append(user_input[0][i])

        predicted_fuel_efficiency = regression_model.predict([user_input_encoded])

        # Find recommended cars based on predicted fuel efficiency
        threshold = predicted_fuel_efficiency[0]
        recommended_cars = df[df['Fuel Efficiency'] >= threshold].head(3)

        return render_template('index.html', tables=[recommended_cars.to_html(classes='data', header="true")])

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
