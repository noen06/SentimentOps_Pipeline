# check_topic_prior.py
import pandas as pd

df = pd.read_csv("dataset/merged_twitter_sentiment.csv")
df = df[df["sentiment"] != "Irrelevant"]
print(pd.crosstab(df["topic"], df["sentiment"], normalize="index").round(3))