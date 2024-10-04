import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load your data here
data_dict = pickle.load(open('data.pickle', 'rb'))

# Print data structure for debugging
print("Data Dictionary:", data_dict)
print("Number of data samples:", len(data_dict['data']))
print("Number of labels:", len(data_dict['labels']))

# Ensure the data is consistent
data = []
labels = []

# Capture the expected length based on your dataset
expected_length = 84  # Update this based on the expected number of features

for i, item in enumerate(data_dict['data']):
    if len(item) == expected_length:
        data.append(item)
        labels.append(data_dict['labels'][i])
    else:
        print(f"Skipping item {i} due to inconsistent length: {len(item)}")

# Convert lists to numpy arrays
data = np.asarray(data)
labels = np.asarray(labels)

# Print the lengths of data and labels after filtering
print("Filtered data samples:", len(data))
print("Filtered labels:", len(labels))

# Proceed with train/test split if we have data
if len(data) == 0 or len(labels) == 0:
    print("Error: No valid samples to train on.")
else:
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, shuffle=True, stratify=labels)

    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)

    print('{}% of samples were classified correctly!'.format(score * 100))

    # Save model and labels dictionary together
    labels_dict = {i: str(label) for i, label in enumerate(np.unique(labels))}

    with open('model.p', 'wb') as f:
        pickle.dump({'model': model, 'labels_dict': labels_dict}, f)