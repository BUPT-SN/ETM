# -*- coding: utf-8 -*-
import pandas as pd

author_profiles_path = "./pandora_comments/author_profiles.csv"
comments_path = "./pandora_comments/all_comments_since_2015.csv"

author_profiles = pd.read_csv(author_profiles_path)

user_to_mbti = {row['author']: row['mbti'].upper() for index, row in author_profiles.iterrows() if isinstance(row['mbti'], str)}


comments = pd.read_csv(comments_path)

comments = comments[comments['author'].isin(user_to_mbti.keys())]

grouped_comments = comments.groupby('author')['body'].apply(lambda x: '|||'.join(x.dropna())).reset_index()

grouped_comments['type'] = grouped_comments['author'].map(user_to_mbti)

result = grouped_comments[['type', 'body']]
result.columns = ['type', 'posts']

output_path = "./pandora_comments/processed_comments.csv"

result.to_csv(output_path, index=False)

print("save to:", output_path)


