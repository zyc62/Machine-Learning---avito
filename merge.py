import pandas as pd
text_data = pd.read_csv('train.csv')
image_data = pd.read_csv('features.csv')

data_merge = pd.merge(text_data,image_data,on = 'id')
df = pd.DataFrame(data_merge)
df.to_csv('demo10100.csv')
