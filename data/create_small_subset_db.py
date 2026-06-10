import pandas as pd

df = pd.read_csv('/Users/maayanmor/Desktop/קורסים תואר שני טכניון/מערכות סוכני בינה מלאכותית/individual_assignment/data/medium-english-50mb.csv')
small_df = df.sample(n=5, random_state=42)
small_df.to_csv('/Users/maayanmor/Desktop/קורסים תואר שני טכניון/מערכות סוכני בינה מלאכותית/individual_assignment/data/medium-english-50mb-small.csv', index=False)