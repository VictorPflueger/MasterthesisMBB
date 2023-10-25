### SHAP

## Suppress warnings (optional) and import libraries
# Import the 'warnings' library for suppressing warnings
import warnings
# Set the filter to ignore all warnings
warnings.filterwarnings("ignore")
# Import dataset california housing from sklearn
from sklearn.datasets import fetch_california_housing
# Import utility for splitting datasets
from sklearn.model_selection import train_test_split
# Import mean_squared_error to evaluate regression model
from sklearn.metrics import mean_squared_error
# Import mean_absolute_error to evaluate regression model
from sklearn.metrics import mean_absolute_error
# Import sqrt function to compute square root
from math import sqrt
# Import Random Forest Regressor model
from sklearn.ensemble import RandomForestRegressor
# Import SHAP for explainable AI
import shap
# Import numpy for numerical operations
import numpy as np
# Import pandas for data manipulation
import pandas as pd
# Import matplotlib for plotting
import matplotlib.pyplot as plt
# Import openai for GPT-4 integration
import openai

## Data pre-processing
# Fetch the california housing dataset and return it as pandas DataFrame
california_housing = fetch_california_housing(as_frame = True)
# Split the dataset into features (X) and target (y)
X = california_housing['data']
y = california_housing['target']
# Split the dataset into training and testing sets, using 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## Define and initialize random forest regressor model and training
# Initialize the Random Forest Regressor model
model = RandomForestRegressor()
# Train the model using the training data
model.fit(X_train, y_train)

## Evaluate model
# Make predictions using the test data
y_pred = model.predict(X_test)
# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
# Calculate Root Mean Squared Error
rmse = sqrt(mse)
# Format the calculated metrics for printing
MSE = f"MSE: {round(mse, 3)}"
MAE = f"MAE: {round(mae, 3)}"
RMSE = f"RMSE: {round(rmse, 3)}"
# Create a list of model performance metrics
model_values = [MSE, MAE, RMSE]
# Print the model performance metrics
print("Model performance metrics:\n" + "\n".join(model_values) + "\n")

## Define and initialize SHAP explanation
# Initialize the SHAP explainer using the model's predict function and test data
explainer = shap.Explainer(model.predict, X_test)
# Compute the SHAP values for the test data
shap_values = explainer(X_test)
# Compute the mean of absolute SHAP values for each feature
mean_shap_values = np.abs(shap_values.values).mean(0)
# Select a random data point from the test set
random_datapoint =  np.random.randint(0, len(X_test))
# Get the model's prediction for the selected data point
f_x = model.predict([X_test.iloc[random_datapoint]])[0]
# Get the expected prediction value for all possible data points
E_f_x = shap_values.base_values[random_datapoint]
# Create a DataFrame to display the SHAP values, feature values, mean SHAP values and weight for the selected data point
shap_table = pd.DataFrame(
    {'Features': list(X_test.columns),
    'Feature value': list(X_test.iloc[random_datapoint]),
    'Mean shap value': list(mean_shap_values),
    'Weight': list(pd.Series(shap_values.values[random_datapoint], index=X_test.columns))
    })
# Round the values in the DataFrame to 3 decimal places
shap_table = shap_table.round(3)
# Sort the DataFrame based on the mean SHAP values in descending order
shap_table = shap_table.sort_values(by=["Mean shap value"], ascending=False)
# Generate a string representation of mean SHAP values for each feature
mean_shap_str = ", ".join([f"{feat} = {round(val, 3)}" for feat, val in zip(X_test.columns, mean_shap_values)])

## Plotting results
# Generate a waterfall plot for the SHAP values of the selected data point
shap.plots.waterfall(shap_values[random_datapoint], show = False)
# Set the title for the plot
plt.title(f"Mean SHAP values: {mean_shap_str}", size=12)
# Display the plot
plt.show()

## GPT-4 prompting
# Define prompts based on the defined prompt patterns
prompt_a = f"Act as a machine learning practitioner. Provide an explanation of the explainable artificial intelligence (XAI) method 'Shapley Additive Explanation (SHAP)'."
prompt_b = f"Act as a machine learning practitioner. Within the scope of a random forest regressor model, the XAI method SHAP was used to explain one instance result of the sklearn dataset california housing. Consider these specifications: The f(x) of the prediction {f_x}, the E[f(x)] of the prediction {E_f_x}, the feature names {X_test.columns}, the feature instance values {X_test.iloc[random_datapoint]}, the obtained mean shap values {mean_shap_str} and the obtained weights {pd.Series(shap_values.values[random_datapoint], index=X_test.columns)}. Provide an overview about all given information."
prompt_c = f"Act as a machine learning practitioner. Within the scope of a random forest regressor model, the XAI method SHAP was used to explain one instance result of the sklearn dataset california housing. Consider these specifications: The f(x) of the prediction {f_x}, the E[f(x)] of the prediction {E_f_x}, the feature names {X_test.columns}, the feature instance values {X_test.iloc[random_datapoint]}, the obtained mean shap values {mean_shap_str} and the obtained weights {pd.Series(shap_values.values[random_datapoint], index=X_test.columns)}. Point out three anomalous values in instance values and explain why they are anomalous, considering your domain knowledge about the california housing dataset and or AI systems."
prompt_d = f"Act as a machine learning practitioner. Within the scope of a random forest regressor model, the XAI method SHAP was used to explain one instance result of the sklearn dataset california housing. Consider these specifications: The f(x) of the prediction {f_x}, the E[f(x)] of the prediction {E_f_x}, the feature names {X_test.columns}, the feature instance values {X_test.iloc[random_datapoint]}, the obtained mean shap values {mean_shap_str} and the obtained weights {pd.Series(shap_values.values[random_datapoint], index=X_test.columns)}. Point out three potential contrastive values in the instance data or feature weights and explain why and to what different prediction they could lead, considering your domain knowledge about the california housing dataset and or AI systems."
# Set the API key for the OpenAI API
openai.api_key = "XXX"
# Define a function to get the completion response from GPT-4 based on the defined modifications
def get_completion(prompt, model="gpt-4"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
    model=model,
    messages=messages,
    temperature=0,
    top_p=1
    )
    return response.choices[0].message["content"]
# Call and print the GPT-4 completions for the defined prompts
print("\n\nGPT - 4 completions:\n\nSHAP - Method:\n" + get_completion(prompt_a) +  "\n\nInformation overview:\n" + get_completion(prompt_b) +  "\n\nAnomalous values:\n" + get_completion(prompt_c) + "\n\nContrastive assessment:\n" + get_completion(prompt_d))