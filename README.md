# Stock-Price-Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
#### Problem Statement

The aim of this experiment is to develop a Recurrent Neural Network (RNN) model to learn sequential patterns from time-series data and make accurate future predictions.

#### Dataset

A time-series dataset is used, where each sample represents sequential data points (such as stock prices, temperature readings, or sensor signals).
The data is preprocessed and divided into training and testing sets to train and evaluate the RNN model effectively.


## Design Steps
### Step 1: 
Import necessary libraries and load the dataset.
### Step 2: 
Preprocess the data and create training and testing sets.
### Step 3: 
Define the RNN model architecture.
### Step 4: 
Initialize loss function, optimizer, and training parameters.
### Step 5: 
Train the model using the training data.
### Step 6: 
Evaluate the model performance on test data.
### Step 7: 
Visualize and analyze the training loss.
## Program
#### Name:RANJIT R
#### Register Number:212224240131

```

# Code to define RNNModel (from __Jiv8IOSbaY)
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.rnn(x, h0)

        # Last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Code to instantiate model, criterion, optimizer (from 9e5rWBONTF9t)
model = RNNModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Original training loop code
epochs = 20

train_losses = [] # Initialize list to store losses

for epoch in range(epochs):
    for x_batch, y_batch in train_loader:
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_losses.append(loss.item()) # Append loss after each epoch
    print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")


```

## Output

### True Stock Price, Predicted Stock Price vs time

<img width="1252" height="792" alt="image" src="https://github.com/user-attachments/assets/437a720d-a964-4abc-a089-822ff36adae1" />


### Predictions 

<img width="488" height="94" alt="image" src="https://github.com/user-attachments/assets/f0baa975-2a92-47e1-9aa9-8ecf3f3b97bb" />


## Result
The Recurrent Neural Network (RNN) model was successfully trained using the given time-series dataset. The training loss gradually decreased over the epochs, showing that the model effectively learned the sequential patterns in the data and produced accurate predictions for future values.

