import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Graph 1: Overall Question Distribution (Pie Chart) ---

# Data for the pie chart
labels_pie = ['Integrator Theory (N=19)', 'Fundamental Concepts (N=5)', 'General Op-Amp (N=3)', 'Analysis & Design (N=3)', 'Non-Lab (N=1)']
sizes = [19, 5, 3, 3, 1]
# Explode the largest slice (Integrator Theory) slightly
explode = (0.1, 0, 0, 0, 0)  
colors = ['#66b3ff', '#99ff99', '#ffcc99', '#ff9999', '#c2c2f0']

# Create the figure
plt.figure(figsize=(10, 7))
# Create the pie chart
plt.pie(sizes, labels=labels_pie, colors=colors, autopct='%1.1f%%',
        shadow=False, startangle=140)
plt.title('Graph 1: Overall Question Distribution (N=31)', pad=20)
# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')  
# Save the figure
plt.savefig('question_distribution_pie_chart.png')
print("Graph 1 saved as 'question_distribution_pie_chart.png'")


# --- Graph 2: Student Satisfaction by Topic (100% Stacked Bar Chart) ---

# Raw data based on your summary
categories = ['Integrator (N=19)', 'Gen. Op-Amp (N=3)', 'Fundamentals (N=5)', 'Analysis (N=3)', 'Non-Lab (N=1)']
data = {
    'Helpful': [14, 0, 4, 2, 0],
    'Mixed': [4, 2, 0, 0, 1],
    'Not Helpful': [0, 0, 0, 1, 0],
    'No Feedback': [1, 1, 1, 0, 0]
}

# Convert to a pandas DataFrame for easier handling
df = pd.DataFrame(data, index=categories)

# Calculate percentages for the 100% stacked bar
# We divide each row by its sum, then multiply by 100
df_percent = df.apply(lambda x: x / x.sum() * 100, axis=1)

# Plotting
# Create a new figure
plt.figure(figsize=(12, 8))
# Plot the 100% stacked bar chart
ax = df_percent.plot(kind='bar', stacked=True, colormap='viridis_r', figsize=(12, 8))

# Set labels and title
plt.title('Graph 2: Student Satisfaction by Topic', pad=20)
plt.ylabel('Percentage of Responses (%)')
plt.xlabel('Question Category')
# Rotate x-axis labels for better readability
plt.xticks(rotation=15, ha='right')  
plt.yticks(np.arange(0, 101, 20))
# Place the legend outside the plot
plt.legend(title='Student Feedback', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to prevent the legend from being cut off
plt.tight_layout()  
# Save the figure
plt.savefig('satisfaction_by_topic_stacked_bar.png')
print("Graph 2 saved as 'satisfaction_by_topic_stacked_bar.png'")