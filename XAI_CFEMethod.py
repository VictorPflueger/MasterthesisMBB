###CFE

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
from sklearn.metrics import precision_score, recall_score, accuracy_score
# Import data utilities from PyTorch
from torch.utils.data import DataLoader, TensorDataset
# Import numpy for numerical operations
import numpy as np
# Import random module
import random
# Import matplotlib for plotting
import matplotlib.pyplot as plt
# Import openai for GPT-4 integration
import openai

## Define dataset attributes
# Defining the class names of the dataset
class_names = [
    'Cultivar 1', 'Cultivar 2', 'Cultivar 3'
]
# Defining the feature names of the dataset
feature_names = [
    'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
    'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
    'Color intensity', 'Hue', 'OD280/OD315', 'Proline'
]

## Data pre-processing
# Load the wine dataset
wine = datasets.load_wine()
# Split the dataset into features (X) and target (y)
X = wine.data
y = wine.target
# Initialize the StandardScaler
scaler = StandardScaler()
# Scale the features
X = scaler.fit_transform(X)
# Split the dataset into training and testing sets, using 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
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
        self.fc1 = nn.Linear(13, 256)
        # Fully connected layer 2
        self.fc2 = nn.Linear(256, 128)
        # Fully connected layer 3
        self.fc3 = nn.Linear(128, 64)
        # Fully connected layer 4
        self.fc4 = nn.Linear(64, 32)
        # Fully connected layer 5
        self.fc5 = nn.Linear(32, 3)

    def forward(self, x):
        # Apply ReLU activation after layer 1
        x = torch.relu(self.fc1(x))
        # Apply ReLU activation after layer 2
        x = torch.relu(self.fc2(x))
        # Apply ReLU activation after layer 3
        x = torch.relu(self.fc3(x))
        # Apply ReLU activation after layer 4
        x = torch.relu(self.fc4(x))
        # Output from layer 5
        x = self.fc5(x)
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

## Define and initialize CFE explanation
# Define a function to compute the counterfactual explanation for a given data point
def counterfactual_explanation(model, data, target_label, lr=0.1, lambda_=0.1, steps=500):
    # Convert the data into a PyTorch tensor, allowing gradient computation on it
    counterfactual = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    # Define the optimizer to adjust the counterfactual; in this case, the Adam optimizer is used
    optimizer = optim.Adam([counterfactual], lr=lr)
    # Loop for a specified number of steps to refine the counterfactual
    for step in range(steps):
        # Get the model's output for the current counterfactual
        model_output = model(counterfactual.unsqueeze(0))
        # Compute the loss. It has two parts: 
        # 1) Negative of the model's confidence in the target label for the counterfactual
        # 2) Regularization term to ensure the counterfactual is close to the original data
        loss = -model_output[0, target_label] + lambda_ * torch.norm(counterfactual - torch.tensor(data, dtype=torch.float32))
        # Zero out any gradients from the previous loop iteration
        optimizer.zero_grad()
        # Compute the gradients of the loss with respect to the counterfactual
        loss.backward()
        # Update the counterfactual using the computed gradients
        optimizer.step()
        # If the model's prediction for the counterfactual matches the target label, exit the loop
        if model(counterfactual.unsqueeze(0)).argmax().item() == target_label:
            break
    # Return the final counterfactual as a numpy array, detaching it from the computational graph
    return counterfactual.detach().numpy()
# Define a function to compute the explanations based on the differences between original and counterfactual data
def explain_counterfactual(original_data, counterfactual_data):
    # Calculate the differences between the original data and counterfactual
    changes = counterfactual_data - original_data
    explanations = []
    # Loop through each feature to generate explanations
    for i in range(len(changes)):
        if changes[i] > 0:
            explanations.append(f"Increase {feature_names[i]} by {changes[i]:.2f}")
        elif changes[i] < 0:
            explanations.append(f"Decrease {feature_names[i]} by {-changes[i]:.2f}")
    # Return the list of explanations
    return explanations
# Take a random sample from the test dataset
for i in random.sample(range(len(X_test)), 1):
    test_data = X_test[i]
    test_label = y_test[i]
    target_label = 0
    # Set the target label to the opposite of the current test label
    if test_label == 0:
        target_label = 1
    elif test_label == 1:
        target_label
    # Get the counterfactual data for the test data point
    counterfactual_data = counterfactual_explanation(model, test_data, target_label)
    # Get the names of the original and counterfactual labels
    original_label_name = class_names[test_label]
    counterfactual_label_name = class_names[model(torch.FloatTensor(counterfactual_data).unsqueeze(0)).argmax().item()]
    # Compute the explanations for the changes in the counterfactual data
    explanations = explain_counterfactual(test_data, counterfactual_data)

## Plotting results
# Generate an array of sequential numbers starting from 0 up to the length of feature_names (exclusive)
x = np.arange(len(feature_names))
# Set the width of each bar in the bar chart to 0.35 units
width = 0.35
# Create a new figure and axis for plotting
fig, ax = plt.subplots()
# Plot a bar chart on the axis using test_data. Bars are shifted to the left by half the width
ax.bar(x - width/2, test_data, width, label='Original Data')
# Plot a bar chart on the axis using counterfactual_data. Bars are shifted to the right by half the width
ax.bar(x + width/2, counterfactual_data, width, label='Counterfactual Data')
# Set the label for the y-axis to 'Feature Value'
ax.set_ylabel('Feature Value')
# Set the tick positions on the x-axis to match the positions of the bars
ax.set_xticks(x)
# Label the ticks on the x-axis using feature_names
ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=10)
# Add a legend to the plot to differentiate between Original Data and Counterfactual Data bars
ax.legend()
# Set the title for the plot, using formatted strings to insert original and counterfactual label names
plt.title(f'Original Label: {original_label_name}, Counterfactual Label: {counterfactual_label_name}')
# Adjust the layout of the plot to ensure that all elements fit and are displayed correctly
plt.tight_layout()
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
prompt_a = f"Act as a machine learning practitioner. Provide an explanation of the explainable artificial intelligence (XAI) method 'Counterfactual Explanation (CFE)'."
prompt_b = f"Act as a machine learning practitioner. Within the scope of a deep learning classification model, the XAI method CFE was used to explain one instance result of the sklearn dataset wine. Consider these specifications: The class names {class_names}, the feature names {feature_names}, the original class {original_label_name}, the counterfactual class {counterfactual_label_name}, the standard scaled data of the original class {test_data} and the data of the counterfactual class {counterfactual_data}. Provide an overview about all given information."
prompt_c = f"Act as a machine learning practitioner. Within the scope of a deep learning classification model, the XAI method CFE was used to explain one instance result of the sklearn dataset wine. Consider these specifications: The class names {class_names}, the feature names {feature_names}, the original class {original_label_name}, the counterfactual class {counterfactual_label_name}, the standard scaled data of the original class {test_data} and the data of the counterfactual class {counterfactual_data}. Point out three anomalous values in the instance data or counterfactual data and explain why they are anomalous, considering your domain knowledge about the wine dataset and or AI systems."
prompt_d = f"Act as a machine learning practitioner. Within the scope of a deep learning classification model, the XAI method CFE was used to explain one instance result of the sklearn dataset wine. Consider these specifications: The class names {class_names}, the feature names {feature_names}, the original class {original_label_name}, the counterfactual class {counterfactual_label_name}, the standard scaled data of the original class {test_data} and the data of the counterfactual class {counterfactual_data}. Point out three contrastive values in the instance data or counterfactual data and explain why they are leading to a different prediction, considering your domain knowledge about the wine dataset and or AI systems."
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
print("\n\nGPT - 4 completions:\n\nCFE - Method:\n" + get_completion(prompt_a) +  "\n\nInformation overview:\n" + get_completion(prompt_b) +  "\n\nAnomalous values:\n" + get_completion(prompt_c) + "\n\nContrastive assessment:\n" + get_completion(prompt_d))