# %% [markdown]
# # 协同过滤推荐
# 

# %% [markdown]
# ## 导入 库文件

# %%
import pandas as pd
import numpy as np

# %%
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
reader = Reader()
ratings = pd.read_csv('../../src/Dataset/Datasets For Collaborative Filtering/ratings_small.csv')
ratings.head()

# %%
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# %%
svd = SVD()

# %%
cross_validate(svd, data, measures=['RMSE', 'MAE'],cv=10)

# %%
trainset = data.build_full_trainset()
svd.fit(trainset)

# %%
svd.predict(1, 302, 3)


