"""import matplotlib.pyplot as plt

# Categories and values as percentages
categories = [
    'Nullstellen', 'Umkehrfunktion', 'Ableitungen',
    'Integrale', 'Polygone', 'Dreiecke',
    'Determinanten', 'Orthogonalisierte Vektoren'
]
values = [16.8, 17.25, 4.25, 13.75, 15.75, 0.0, 2, 5.4]

# Create the bar chart
plt.bar(categories, values, color='lightblue')

# Add title and labels
plt.title('Performanz (GPT2)')
plt.xlabel('Kategorien')
plt.ylabel('Genauigkeit')

# Set y-axis limit from 0 to 100
plt.ylim(0, 100)

# Rotate category labels for better readability
plt.xticks(rotation=45, ha='right')

# Display the chart with better spacing
plt.tight_layout()

# Show the chart
plt.show()"""

import matplotlib.pyplot as plt
import numpy as np

# Categories and two sets of values
categories = [
    'Nullstellen', 'Umkehrfunktion', 'Ableitungen',
    'Integrale', 'Polygone', 'Dreiecke',
    'Determinanten', 'Orthogonalisierte Vektoren'
]

# Values for two different datasets (example values in percentage)
values_set1 = [27.5, 28.2, 18.0, 21.7, 22.9, 6.7, 9.5, 16.9]
values_set2 = [32.8, 35.0, 27.7, 26.3, 28.5, 13.1, 18.0, 23.5]

# Set the width of the bars
bar_width = 0.35
# Generate the position for each bar on the x-axis
index = np.arange(len(categories))

# Create a larger figure
plt.figure(figsize=(7, 5))  # Width: 12 inches, Height: 6 inches

# Create the grouped bar chart
plt.bar(index, values_set1, bar_width, label='GPT2', color='blue')
plt.bar(index + bar_width, values_set2, bar_width, label='GPT NEO', color='darkviolet')

# Add title and labels
plt.title('Problem-Antwort')
plt.xlabel('Kategorien')
plt.ylabel('Genauigkeit in %')

# Set y-axis limit from 0 to 100
plt.ylim(0, 100)

# Set the position of the x-ticks and their labels
plt.xticks(index + bar_width / 2, categories, rotation=45, ha='right')

# Add a legend to distinguish between the datasets
plt.legend()

# Display the chart with better spacing
plt.tight_layout()

# Show the chart
plt.show()
