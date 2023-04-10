import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Example ground truth and predicted labels
y_true = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0]

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Define labels for rows and columns
labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

# Reshape confusion matrix into a 1D array
cm_1d = cm.ravel()

# Calculate percentages for each cell in the confusion matrix
cm_perc = cm_1d / np.sum(cm_1d)

# Reshape percentages back into a 2D array
cm_perc = cm_perc.reshape((2, 2))

# Create heatmap using seaborn library
sns.heatmap(cm_perc, annot=True, cmap="Blues", fmt='.2%')

# Add labels to the heatmap
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.xticks(np.arange(2)+0.5, ['Negative', 'Positive'])
plt.yticks(np.arange(2)+0.5, ['Negative', 'Positive'])
plt.title('Confusion Matrix')

# Show the plot
plt.show()
