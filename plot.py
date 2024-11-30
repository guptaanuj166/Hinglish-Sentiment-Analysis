import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("train_data.csv")

# Count the occurrences of each value in the column
value_counts = df['label'].value_counts()

# Create the pie chart
plt.figure(figsize=(8, 8))
pie = value_counts.plot.pie(
    autopct='%1.1f%%',
    startangle=90,
    colors=plt.cm.Set2.colors,
    labels=["Negative", "Neutral", "Positive"]  # Disable default labels for a cleaner look
)

# Add legend

plt.title('Distribution of Categories')
plt.ylabel('')  # Hide the y-label for a cleaner look
plt.savefig("pie_plot.png")
plt.close()