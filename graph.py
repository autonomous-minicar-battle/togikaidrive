import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data from the CSV file
folder = "records"
csv_files = []
for file in os.listdir(folder):
    if file.endswith(".csv"):
        csv_files.append(file)
print(csv_files)
csv_file = input("csvファイル名を入力してください: ")
csv_path = os.path.join(folder, csv_file)
#csv_file = 'record_20240622_040833.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_path)

# Check the structure of the dataframe
print(df.head())

# Extract columns
timestamps = df['Tstamp']
str_values = df['Str']
thr_values = df['Thr']
sensor_values = df.drop(columns=['Tstamp', 'Str', 'Thr'])

# Plot the data
plt.figure(figsize=(14, 8))

# Plot Str and Thr
plt.subplot(2, 1, 1)
plt.plot(timestamps, str_values, label='Str', color='blue')
plt.plot(timestamps, thr_values, label='Thr', color='red')
plt.xlabel('Timestamp')
plt.ylabel('Output Values')
plt.legend()
plt.title('Output Values (Str and Thr) Over Time')

# Plot sensor values
plt.subplot(2, 1, 2)
for column in sensor_values.columns:
    plt.plot(timestamps, sensor_values[column], label=column)
plt.xlabel('Timestamp')
plt.ylabel('Sensor Input Values')
plt.legend()
plt.title('Sensor Input Values Over Time')
plt.tight_layout()

# Show the plot
print("\nグラフを表示します, バツボタンで終了と画像の保存ができます。")
plt.show()

# Save the plot as an image
png =  csv_path.replace('.csv', '.png')
print("\nSaving this plot >>>", png)
plt.savefig(csv_path.replace('.csv', '.png'))

