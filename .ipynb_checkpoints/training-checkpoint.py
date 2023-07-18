import torch
import random
import numpy as np
from rnn import RNNNet
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from dataset import StimuliDataset

import torch.nn as nn

# Parameters


train_samples = 6400*2
test_samples = 6400
hidden_size = 128
input_size = 13
output_size = 6
dt = 0.1
tau = 100
batch_size = 32
num_epochs = 20
model_path = './model.pt'

# Create an instance of the StimuliDataset for training data
train_dataset = StimuliDataset(train_samples)
# Create an instance of the StimuliTestDataset for testing data
test_dataset = StimuliDataset(test_samples)
rnn_activities = []


# Instantiate the network and print information
net = RNNNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dt=dt, tau = tau)
print(net)

# Use Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=0.3)
criterion = nn.CrossEntropyLoss()

running_loss = 0
running_acc = 0

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the network to the device
net = net.to(device)

# Create a DataLoader for batch loading and shuffling for training data
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# Create a DataLoader for batch loading and shuffling for testing data
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for epoch in range(num_epochs):
    running_loss_train = 0.0
    running_acc_train = 0.0
    num_batches_train = 0
    for batch_idx, (stimulus, labels) in enumerate(train_dataloader):
       #  print(labels.shape)
        # print(labels)
        labels = torch.tensor(labels.squeeze(-1), dtype=torch.int64)
        #labels = nn.functional.one_hot(torch.tensor(labels.squeeze(-1), dtype=torch.int64), num_classes=6)
        stimuli = np.stack([stimulus] * 3, axis=1)
        stop_cue = np.zeros((32,1,input_size))
        
        inputs  = np.concatenate((stimuli, stop_cue), axis=1)
        
        #labels = labels.type(torch.LongTensor)   # casting to long
        inputs = torch.from_numpy(inputs).float().to(device)
        labels = labels.float().to(device)  # Convert labels to long and move to device


        optimizer.zero_grad()
        output, hidden_dyns = net(inputs.permute(1, 0, 2))
        rnn_activities.append(hidden_dyns)
        #output = torch.tensor(output, dtype = torch.int32)

        predicted_labels = torch.argmax(output, dim=1)
        correct_predictions = (predicted_labels == labels).float()
        acc = torch.mean(correct_predictions)

        labels=labels.to(torch.int64)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss_train += loss.item()
        running_acc_train += acc
        num_batches_train += 1

    average_loss_train = running_loss_train / num_batches_train
    average_acc_test = running_acc_train / num_batches_train
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss_train:.4f}, Accuracy: {average_acc_test:.4f}')    
    # Evaluation phase (using the testing data)
    net.eval()  # Set the network to evaluation mode
    with torch.no_grad():  # Disable gradient computation for evaluation
        running_loss_test= 0
        num_batches_test= 0
        running_acc_test = 0.0

        for batch_idx, (stimulus, labels) in enumerate(test_dataloader):
            labels = torch.tensor(labels.squeeze(-1), dtype=torch.int64)
            #labels = nn.functional.one_hot(torch.tensor(labels.squeeze(-1), dtype=torch.int64), num_classes=6)
            stimuli = np.stack([stimulus] * 3, axis=1)
            stop_cue = np.zeros((32,1,input_size))

            inputs  = np.concatenate((stimuli, stop_cue), axis=1)

            #labels = labels.type(torch.LongTensor)   # casting to long
            inputs = torch.from_numpy(inputs).float().to(device)
            labels = labels.float().to(device)  # Convert labels to long and move to device

            output, hidden_dyns = net(inputs.permute(1, 0, 2))
            rnn_activities.append(hidden_dyns)
            #output = torch.tensor(output, dtype = torch.int32)
            predicted_labels = torch.argmax(output, dim=1)
            correct_predictions = (predicted_labels == labels).float()
            acc = torch.mean(correct_predictions)

            #acc = (torch.argmax(output) == torch.argmax(labels)).sum()/stimulus.shape[0]

            labels=labels.to(torch.int64)
            loss = criterion(output, labels)
            running_loss_test += loss.item()
            running_acc_test += acc
            num_batches_test += 1

        average_loss_test = running_loss_test / num_batches_test
        average_acc_test = running_acc_test / num_batches_test
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss_test:.4f}, Accuracy: {average_acc_test:.4f}')
        
# Save the model
torch.save(net.state_dict(), model_path)
