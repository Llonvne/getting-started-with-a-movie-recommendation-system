# %% [markdown]
# # 基于人口过滤器的推荐程序

# %% [markdown]
# ## 导入库文件

# %%
import pandas as pd
import numpy as np

# %% [markdown]
# ## 导入数据集

# %%
df1 = pd.read_csv('../../src/Dataset/Datasets For Demographic Filtering and Content Based Filtering/tmdb_5000_credits.csv')
df2 = pd.read_csv('../../src/Dataset/Datasets For Demographic Filtering and Content Based Filtering/tmdb_5000_movies.csv')

# %% [markdown]
# ## 链接两个数据集合 （ID）

# %%
df1.columns = ['id','tittle','cast','crew']
df2= df2.merge(df1,on='id')

# %% [markdown]
# ## 计算平均分数

# %%
C= df2['vote_average'].mean()
print("Vote Average is ",C)

# %% [markdown]
# ## 确定最低票数要求

# %%
m= df2['vote_count'].quantile(0.9)
print("the minimum votes requires is",m)

# %% [markdown]
# ## 筛选出符合要求的电影

# %%
q_movies = df2.copy().loc[df2['vote_count'] >= m]
q_movies.shape

# %% [markdown]
# ## 定义比分计算函数

# %%
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C) 

# %% [markdown]
# ## 为过滤出来的电影评分

# %%
# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

# %% [markdown]
# ## 打印表格

# %%
#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 15 movies
q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)

# %% [markdown]
# ## 画图

# %%
pop= df2.sort_values('popularity', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(6),pop['popularity'].head(6), align='center',
        color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")


