import kagglehub
df = kagglehub.load_dataset("kagglehub.KaggleDatasetAdapter.PANDAS","wcukierski/enron-email-dataset","emails.csv")

print(df.head())
