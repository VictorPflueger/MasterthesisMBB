###LIME

## Suppress warnings (optional) and import libraries
# Import the 'warnings' library for suppressing warnings
import warnings
# Set the filter to ignore all warnings
warnings.filterwarnings("ignore")
# Import PyTorch library
import torch
# Import neural network module from PyTorch
import torch.nn as nn
# Import optimizer module from PyTorch
import torch.optim as optim
# Import datasets from scikit-learn
from sklearn import datasets
# Import train-test split function
from sklearn.model_selection import train_test_split
# Import standard scaler for data normalization
from sklearn.preprocessing import StandardScaler
# Import model performance metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score
# Import data utilities from PyTorch
from torch.utils.data import DataLoader, TensorDataset
# Import LIME explainer for tabular data
from lime.lime_tabular import LimeTabularExplainer
# Import matplotlib for plotting
import matplotlib.pyplot as plt
# Import numpy for numerical operations
import numpy as np
# Import pandas for data manipulation
import pandas as pd
# Import openai for GPT-4 integration
import openai

## Define dataset attributes
# Defining the basic feature names of the dataset
feature_names = [
    "Elevation (meters)",
    "Aspect (degrees azimuth)",
    "Slope (degrees)",
    "Horizontal Distance to Hydrology (meters)",
    "Vertical Distance to Hydrology (meters)",
    "Horizontal Distance to Roadways (meters)",
    "Hillshade 9am (0-255 index)",
    "Hillshade Noon (0-255 index)",
    "Hillshade 3pm (0-255 index)",
    "Horizontal Distance to Fire Points (meters)"
]
# Defining the wilderness areas of the dataset
wilderness_areas = [
    "Rawah Wilderness Area",
    "Neota Wilderness Area",
    "Comanche Peak Wilderness Area",
    "Cache la Poudre Wilderness Area"
]
# Defining the soil types areas of the dataset
soil_types = [
    "Cathedral family - Rock outcrop complex, extremely stony",
    "Vanet - Ratake families complex, very stony",
    "Haploborolis - Rock outcrop complex, rubbly",
    "Ratake family - Rock outcrop complex, rubbly",
    "Vanet family - Rock outcrop complex complex, rubbly",
    "Vanet - Wetmore families - Rock outcrop complex, stony",
    "Gothic family",
    "Supervisor - Limber families complex",
    "Troutville family, very stony",
    "Bullwark - Catamount families - Rock outcrop complex, rubbly",
    "Bullwark - Catamount families - Rock land complex, rubbly",
    "Legault family - Rock land complex, stony",
    "Catamount family - Rock land - Bullwark family complex, rubbly",
    "Pachic Argiborolis - Aquolis complex",
    "Unspecified in the USFS Soil and ELU Survey",
    "Cryaquolis - Cryoborolis complex",
    "Gateview family - Cryaquolis complex",
    "Rogert family, very stony",
    "Typic Cryaquolis - Borohemists complex",
    "Typic Cryaquepts - Typic Cryaquolls complex",
    "Typic Cryaquolls - Leighcan family, till substratum complex",
    "Leighcan family, till substratum, extremely bouldery",
    "Leighcan family, till substratum - Typic Cryaquolls complex",
    "Leighcan family, extremely stony",
    "Leighcan family, warm, extremely stony",
    "Granile - Catamount families complex, very stony",
    "Leighcan family, warm - Rock outcrop complex, extremely stony",
    "Leighcan family - Rock outcrop complex, extremely stony",
    "Como - Legault families complex, extremely stony",
    "Como family - Rock land - Legault family complex, extremely stony",
    "Leighcan - Catamount families complex, extremely stony",
    "Catamount family - Rock outcrop - Leighcan family complex, extremely stony",
    "Leighcan - Catamount families - Rock outcrop complex, extremely stony",
    "Cryorthents - Rock land complex, extremely stony",
    "Cryumbrepts - Rock outcrop - Cryaquepts complex",
    "Bross family - Rock land - Cryumbrepts complex, extremely stony",
    "Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony",
    "Leighcan - Moran families - Cryaquolls complex, extremely stony",
    "Moran family - Cryorthents - Leighcan family complex, extremely stony",
    "Moran family - Cryorthents - Rock land complex, extremely stony"
]
# Defining the class names of the dataset
class_names = [
    "Spruce/Fir", 
    "Lodgepole Pine", 
    "Ponderosa Pine", 
    "Cottonwood/Willow", 
    "Aspen", 
    "Douglas-fir", 
    "Krummholz"
    ]

## Data pre-processing
# Load the covertype dataset
covertype = datasets.fetch_covtype()
# Split the dataset into features (X) and target (y)
X = covertype.data
y = covertype.target - 1
# Initialize the StandardScaler
scaler = StandardScaler()
# Scale the features
X_scaled = scaler.fit_transform(X)
# Split the dataset into training and testing sets, using 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Converting numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
# Creating datasets and loaders for batching during training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

## Define and initialize neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Fully connected layer 1
        self.fc1 = nn.Linear(54, 128)
        # Fully connected layer 2
        self.fc2 = nn.Linear(128, 64)
        # Fully connected layer 3
        self.fc3 = nn.Linear(64, 7)
    def forward(self, x):
        # Apply ReLU activation after layer 1
        x = torch.relu(self.fc1(x))
        # Apply ReLU activation after layer 2
        x = torch.relu(self.fc2(x))
        # Output from layer 3
        x = self.fc3(x)
        return x
# Initialize the model
model = Net()
# Define the loss function
criterion = nn.CrossEntropyLoss()
# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

## Define evaluation function
def evaluate_model(model, test_loader):
    # Set the model to evaluation mode
    model.eval()
    # Initialize list to store the true labels
    y_true = []
    # Initialize list to store the predicted labels
    y_pred = []
    # Initialize the validation loss counter to zero
    val_loss = 0
    # Ensure that no gradients are computed during evaluation
    with torch.no_grad():
        # Iterate over batches from the test loader
        for x_batch, y_batch in test_loader:
            # Pass the input batch through the model to get predictions
            outputs = model(x_batch)
            # Compute the loss between predictions and true labels
            loss = criterion(outputs, y_batch)
            # Increment the validation loss
            val_loss += loss.item()
            # Find the class with the highest probability for predictions
            _, preds = torch.max(outputs, dim=1)
            # Add true labels to the y_true list
            y_true.extend(y_batch.tolist())
            # Add predicted labels to the y_pred list
            y_pred.extend(preds.tolist())
    # Compute the average validation loss
    val_loss /= len(test_loader)
    # Compute the accuracy
    accuracy = accuracy_score(y_true, y_pred)
    # Compute the precision
    precision = precision_score(y_true, y_pred, average='weighted')
    # Compute the weighted recall
    recall = recall_score(y_true, y_pred, average='weighted')
    # Return the computed metrics
    return val_loss, accuracy, precision, recall

## Define and initialize model training
# Define the number of epochs
epochs = 100
# List to store training losses for each epoch
train_losses = []
# List to store validation losses for each epoch
val_losses = []
# List to store accuracies for each epoch
accuracy_tracker = []
# List to store precisions for each epoch
precision_tracker = []
# List to store recalls for each epoch
recall_tracker = []
# Loop over the epochs
for epoch in range(epochs):
    # Set the model to training mode
    model.train()
    # Iterate over batches from the train loader
    for i, (x_batch, y_batch) in enumerate(train_loader):
        # Pass the input batch through the model to get predictions
        outputs = model(x_batch)
        # Compute the loss between predictions and true labels
        loss = criterion(outputs, y_batch)
        # Zero out the gradients to prepare for backpropagation
        optimizer.zero_grad()
        # Perform backpropagation based on the loss
        loss.backward()
        # Update the model's parameters
        optimizer.step()
    # Store the last batch's training loss for this epoch
    train_losses.append(loss.item())
    # Evaluate the model on the test set
    val_loss, accuracy, precision, recall = evaluate_model(model, test_loader)
    # Store the validation loss
    val_losses.append(val_loss)
    # Store the accuracy
    accuracy_tracker.append(accuracy)
    # Store the precision
    precision_tracker.append(precision)
    # Store the recall
    recall_tracker.append(recall)
    # Print metrics every 25 epochs
    if epoch % 25 == 0:
        print(f"Epoch [{epoch}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

## Define and initialize LIME explanation 
# Combine the feature names
feature_names += wilderness_areas + soil_types
# Select a random index from the test set
random_idx = np.random.randint(0, X_test.shape[0])
# Retrieve the datapoint associated with the random index
random_datapoint = X_test[random_idx]
# Get the actual class of the randomly selected datapoint
actual_class = class_names[y_test[random_idx]]
# Initialize the LIME explainer
explainer = LimeTabularExplainer(X_train, training_labels=y_train, feature_names=feature_names, class_names=class_names, mode="classification")
# Define a function to get the probability predictions
def predict_proba(input_data):
    # Set the model to evaluation mode
    model.eval()
    # Ensure that no gradients are computed during prediction
    with torch.no_grad():
        # Convert input data to PyTorch tensor
        tensor_input = torch.tensor(input_data, dtype=torch.float32)
        # Get the raw output from the model
        raw_output = model(tensor_input)
        # Convert raw output to probabilities
        probabilities = nn.Softmax(dim=1)(raw_output)
    # Return the probabilities as a numpy array
    return probabilities.numpy()
# Use LIME to explain the prediction for the random datapoint
explanation = explainer.explain_instance(random_datapoint, predict_proba, num_features=len(feature_names))
# Get the index of the highest probability class
predicted_class_index = np.argmax(predict_proba(random_datapoint.reshape(1, -1)))
# Get the class name for the predicted class
predicted_class = class_names[predicted_class_index]
# Get the probabilities as percentages
prediction_probabilities = np.round(predict_proba(random_datapoint.reshape(1, -1)) * 100, 2)
# Create a list of strings representing each class's probability.
prediction_probabilities_str = [
    f"{class_name}: < 0.01 %" if p < 0.01 else f"{class_name}: {p:.2f} %"
    for p, class_name in zip(prediction_probabilities[0], class_names)
]
# Create a DataFrame for feature values
instance_values_df = pd.DataFrame({'Feature': feature_names, 'Instance Value': random_datapoint})
# Get the explanation values from LIME
lime_values = explanation.as_list()
# Convert LIME values to a DataFrame
value_range_df = pd.DataFrame(lime_values, columns=['Value range', 'Weight'])
# Create a copy of the feature values DataFrame
lime_df = instance_values_df
# Define a function to merge the LIME explanation with the feature values.
def merge_on_condition(row, source_df, source_col='Value range', target_col='Weight'):
    # Get the feature name from the row
    feature = row['Feature']
    # Iterate over the LIME values DataFrame
    for index, source_row in source_df.iterrows():
        # Check if the feature name appears in the LIME explanation
        if feature.lower() in source_row[source_col].lower():
            # Return the explanation values for this feature
            return pd.Series([source_row[source_col], source_row[target_col]])
    # Return None if the feature doesn't appear in the explanation
    return pd.Series([None, None])
# Apply the merge function to each row in the DataFrame
lime_df[['Value range', 'Weight']] = lime_df.apply(lambda row: merge_on_condition(row, value_range_df), axis=1)
# Sort the DataFrame by the LIME weight in descending order
lime_df = lime_df.sort_values('Weight', ascending=False)

## Plotting results
# Create a new figure and axis object with a specified size for the LIME plot
fig, ax1 = plt.subplots(figsize=(14, 10))
# Determine the colors of the bars based on their weights (green if positive, red if negative)
colors = ['green' if weight > 0 else 'red' for weight in lime_df['Weight']]
# Create horizontal bars on ax1 using the 'Value range' as the y-values and 'Weight' as the width of bars, colored based on the colors list
bars = ax1.barh(lime_df['Value range'], lime_df['Weight'], color=colors)
# Set the label for the x-axis of ax1
ax1.set_xlabel('Weight')
# Set the label for the y-axis of ax1
ax1.set_ylabel('Value Range')
# Set the title for ax1 using formatted string (which includes predicted and actual class, and prediction probabilities)
ax1.set_title(f"Model predicts: {predicted_class}, Actual class: {actual_class} \n\nPrediction probabilities: {prediction_probabilities_str}", size=12)
# Create a second y-axis on ax1 that shares the same x-axis
ax2 = ax1.twinx()
# Ensure the y-axis limits of ax2 are the same as ax1
ax2.set_ylim(ax1.get_ylim())
# Set y-ticks on ax2 to match the number of rows in lime_df
ax2.set_yticks(np.arange(len(lime_df)))
# Create custom y-tick labels for ax2 displaying instance values and corresponding feature names
ax2_yticklabels = lime_df.apply(lambda row: f"{row['Instance Value']:.4f} ({row['Feature']})", axis=1)
# Set the custom y-tick labels to ax2
ax2.set_yticklabels(ax2_yticklabels)
# Set the label for the y-axis of ax2
ax2.set_ylabel('Instance Value')
# Display the plot created above
plt.show()
# Plot training loss across epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training loss across epochs')
plt.show()
# Plot validation across epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), val_losses)
plt.xlabel('Epoch')
plt.ylabel('Val Loss')
plt.title('Val Loss across epochs')
plt.show()
# Plot accuracy across epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), accuracy_tracker)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy across epochs')
plt.show()
# Plot precision across epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), precision_tracker)
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Precision across epochs')
plt.show()
# Plot recall across epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), recall_tracker)
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Recall across epochs')
plt.show()

## GPT-4 prompting
# Define prompts based on the defined prompt patterns
prompt_a = f"Act as a machine learning practitioner. Provide an explanation of the explainable artificial intelligence (XAI) method 'Local Interpretable Model-Agnostic Explanation (LIME)'."
prompt_b = f"Act as a machine learning practitioner. Within the scope of a deep learning classification model, the XAI method LIME was used to explain one instance result of the sklearn dataset covertype. Consider these specifications: The class names {class_names}, the feature names {feature_names}, the actual class {actual_class}, the predicted class {predicted_class}, the prediction probabilities {prediction_probabilities_str}, the dataframe {instance_values_df} containing the feature names and the instance values and the dataframe {value_range_df} containing the value ranges of the feature names and their corresponding weights. Provide an overview about all given information."
prompt_c = f"Act as a machine learning practitioner. Within the scope of a deep learning classification model, the XAI method LIME was used to explain one instance result of the sklearn dataset covertype. Consider these specifications: The class names {class_names}, the feature names {feature_names}, the actual class {actual_class}, the predicted class {predicted_class}, the prediction probabilities {prediction_probabilities_str}, the dataframe {instance_values_df} containing the feature names and the instance values and the dataframe {value_range_df} containing the value ranges of the feature names and their corresponding weights. Point out three anomalous values in instance values, value ranges and or LIME weights and explain why they are anomalous, considering your domain knowledge about the covertype dataset and or AI systems."
prompt_d = f"Act as a machine learning practitioner. Within the scope of a deep learning classification model, the XAI method LIME was used to explain one instance result of the sklearn dataset covertype. Consider these specifications: The class names {class_names}, the feature names {feature_names}, the actual class {actual_class}, the predicted class {predicted_class}, the prediction probabilities {prediction_probabilities_str}, the dataframe {instance_values_df} containing the feature names and the instance values and the dataframe {value_range_df} containing the value ranges of the feature names and their corresponding weights. Point out three potential contrastive values in the instance data or feature weights and explain why and to what different prediction class they could lead, considering your domain knowledge about the covertype dataset and or AI systems."
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
print("\n\nGPT - 4 completions:\n\nLIME - Method:\n" + get_completion(prompt_a) +  "\n\nInformation overview:\n" + get_completion(prompt_b) +  "\n\nAnomalous values:\n" + get_completion(prompt_c) + "\n\nContrastive assessment:\n" + get_completion(prompt_d))