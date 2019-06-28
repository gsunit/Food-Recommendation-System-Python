# Food-Recmmendation-System-Python



```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```

# Lunchbox App ML Engine

This is the Recommendation Engine that will be used in building the <b>Lunchbox App</b>, a platform for ordering food and keeping track of user expenditure and canteen sales. Regardless of whether or not this is actually implemented in all the canteens of <b>IIT Kanpur</b>, given the potential for frauds & cyber-attacks, I will complete the platform.

Also, <b>I would be open-sourcing the app</b> so that any campus can implement a <b>cash-less & integrated system of ordering food</b> across their whole campus. After all, what good are IITs for if our canteens still keep track of student accounts on paper registers!

## Demographic Filtering

Suggesting the users items that were well-received and are popular among the users, in general. Most trending items and items with the best rating rise to the top and get shortlisted for recommendation.


```python
import pandas as pd 
import numpy as np

# Importing db of food items across all canteens registered on the platform
df1=pd.read_csv('./db/food.csv')
df1.columns = ['food_id','title','canteen_id','price', 'num_orders', 'category', 'avg_rating', 'num_rating', 'tags']

df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>food_id</th>
      <th>title</th>
      <th>canteen_id</th>
      <th>price</th>
      <th>num_orders</th>
      <th>category</th>
      <th>avg_rating</th>
      <th>num_rating</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Lala Maggi</td>
      <td>1</td>
      <td>30</td>
      <td>35</td>
      <td>maggi</td>
      <td>3.9</td>
      <td>10</td>
      <td>veg, spicy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Cheese Maggi</td>
      <td>1</td>
      <td>25</td>
      <td>40</td>
      <td>maggi</td>
      <td>3.8</td>
      <td>15</td>
      <td>veg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Masala Maggi</td>
      <td>1</td>
      <td>25</td>
      <td>10</td>
      <td>maggi</td>
      <td>3.0</td>
      <td>10</td>
      <td>veg, spicy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Veg Maggi</td>
      <td>1</td>
      <td>30</td>
      <td>25</td>
      <td>maggi</td>
      <td>2.5</td>
      <td>5</td>
      <td>veg, healthy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Paneer Tikka</td>
      <td>1</td>
      <td>60</td>
      <td>50</td>
      <td>Punjabi</td>
      <td>4.6</td>
      <td>30</td>
      <td>veg, healthy</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Chicken Tikka</td>
      <td>1</td>
      <td>80</td>
      <td>40</td>
      <td>Punjabi</td>
      <td>4.2</td>
      <td>28</td>
      <td>nonveg, healthy, spicy</td>
    </tr>
  </tbody>
</table>
</div>




```python
# mean of average ratings of all items
C= df1['avg_rating'].mean()

# the minimum number of votes required to appear in recommendation list, i.e, 60th percentile among 'num_rating'
m= df1['num_rating'].quantile(0.6)

# items that qualify the criteria of minimum num of votes
q_items = df1.copy().loc[df1['num_rating'] >= m]

# Calculation of weighted rating based on the IMDB formula
def weighted_rating(x, m=m, C=C):
    v = x['num_rating']
    R = x['avg_rating']
    return (v/(v+m) * R) + (m/(m+v) * C)

# Applying weighted_rating to qualified items
q_items['score'] = q_items.apply(weighted_rating, axis=1)

# Shortlisting the top rated items and popular items
top_rated_items = q_items.sort_values('score', ascending=False)
pop_items= df1.sort_values('num_orders', ascending=False)
```


```python
# Display results of demographic filtering
top_rated_items[['title', 'num_rating', 'avg_rating', 'score']].head()
pop_items[['title', 'num_orders']].head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>num_rating</th>
      <th>avg_rating</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Paneer Tikka</td>
      <td>30</td>
      <td>4.6</td>
      <td>4.288889</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Chicken Tikka</td>
      <td>28</td>
      <td>4.2</td>
      <td>4.013953</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cheese Maggi</td>
      <td>15</td>
      <td>3.8</td>
      <td>3.733333</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>num_orders</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Paneer Tikka</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cheese Maggi</td>
      <td>40</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Chicken Tikka</td>
      <td>40</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Lala Maggi</td>
      <td>35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Veg Maggi</td>
      <td>25</td>
    </tr>
  </tbody>
</table>
</div>



## Content Based Filtering

A bit more personalised recommendation. We will be analysing the past orders of the user and suggesting back those items which are similar.

Also, since each person has a "home canteen", the user should be notified any new items included in the menu by the vendor.

We will be using <b>Count Vectorizer</b> from <b>Scikit-Learn</b> to find similarity between items based on their title, category and tags. To bring all these properties of each item together we create a <b>"soup"</b> of tags. <b>"Soup"</b> is a processed string correspnding to each item, formed using constituent words of tags, tile and category.


```python
# TODO: clean data

# Creating soup string for each item
def create_soup(x):            
    tags = x['tags'].lower().split(', ')
    tags.extend(x['title'].lower().split())
    tags.extend(x['category'].lower().split())
    return " ".join(sorted(set(tags), key=tags.index))

df1['soup'] = df1.apply(create_soup, axis=1)
df1.head(3)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>food_id</th>
      <th>title</th>
      <th>canteen_id</th>
      <th>price</th>
      <th>num_orders</th>
      <th>category</th>
      <th>avg_rating</th>
      <th>num_rating</th>
      <th>tags</th>
      <th>soup</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Lala Maggi</td>
      <td>1</td>
      <td>30</td>
      <td>35</td>
      <td>maggi</td>
      <td>3.9</td>
      <td>10</td>
      <td>veg, spicy</td>
      <td>veg spicy lala maggi</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Cheese Maggi</td>
      <td>1</td>
      <td>25</td>
      <td>40</td>
      <td>maggi</td>
      <td>3.8</td>
      <td>15</td>
      <td>veg</td>
      <td>veg cheese maggi</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Masala Maggi</td>
      <td>1</td>
      <td>25</td>
      <td>10</td>
      <td>maggi</td>
      <td>3.0</td>
      <td>10</td>
      <td>veg, spicy</td>
      <td>veg spicy masala maggi</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(stop_words='english')

# df1['soup']
count_matrix = count.fit_transform(df1['soup'])

# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(count_matrix, count_matrix)

indices_from_title = pd.Series(df1.index, index=df1['title'])
indices_from_food_id = pd.Series(df1.index, index=df1['food_id'])
```


```python
# Function that takes in food title or food id as input and outputs most similar dishes 
def get_recommendations(title="", cosine_sim=cosine_sim, idx=-1):
    # Get the index of the item that matches the title
    if idx == -1 and title != "":
        idx = indices_from_title[title]

    # Get the pairwsie similarity scores of all dishes with that dish
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the dishes based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 10 most similar dishes
    sim_scores = sim_scores[1:3]

    # Get the food indices
    food_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar dishes
    return food_indices
```


```python
df1.loc[get_recommendations(title="Paneer Tikka")]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>food_id</th>
      <th>title</th>
      <th>canteen_id</th>
      <th>price</th>
      <th>num_orders</th>
      <th>category</th>
      <th>avg_rating</th>
      <th>num_rating</th>
      <th>tags</th>
      <th>soup</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Chicken Tikka</td>
      <td>1</td>
      <td>80</td>
      <td>40</td>
      <td>Punjabi</td>
      <td>4.2</td>
      <td>28</td>
      <td>nonveg, healthy, spicy</td>
      <td>nonveg healthy spicy chicken tikka punjabi</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Veg Maggi</td>
      <td>1</td>
      <td>30</td>
      <td>25</td>
      <td>maggi</td>
      <td>2.5</td>
      <td>5</td>
      <td>veg, healthy</td>
      <td>veg healthy maggi</td>
    </tr>
  </tbody>
</table>
</div>



We will now some functions, some of which are utility functions, others are actually the functions which will help get personalised recommendations for current user.


```python
# fetch few past orders of a user, based on which personalized recommendations are to be made
def get_latest_user_orders(user_id, orders, num_orders=3):
    counter = num_orders
    order_indices = []
    
    for index, row in orders[['user_id']].iterrows():
        if row.user_id == user_id:
            counter = counter -1
            order_indices.append(index)
        if counter == 0:
            break
            
    return order_indices

# utility function that returns a DataFrame given the food_indices to be recommended
def get_recomms_df(food_indices, df1, columns, comment):
    row = 0
    df = pd.DataFrame(columns=columns)
    
    for i in food_indices:
        df.loc[row] = df1[['title', 'canteen_id', 'price']].loc[i]
        df.loc[row].comment = comment
        row = row+1
    return df

# return food_indices for accomplishing personalized recommendation using Count Vectorizer
def personalised_recomms(orders, df1, user_id, columns, comment="based on your past orders"):
    order_indices = get_latest_user_orders(user_id, orders)
    food_ids = []
    food_indices = []
    recomm_indices = []
    
    for i in order_indices:
        food_ids.append(orders.loc[i].food_id)
    for i in food_ids:
        food_indices.append(indices_from_food_id[i])
    for i in food_indices:
        recomm_indices.extend(get_recommendations(idx=i))
        
    return get_recomms_df(set(recomm_indices), df1, columns, comment)

# Simply fetch new items added by vendor or today's special at home canteen
def get_new_and_specials_recomms(new_and_specials, users, df1, canteen_id, columns, comment="new/today's special item  in your home canteen"):
    food_indices = []
    
    for index, row in new_and_specials[['canteen_id']].iterrows():
        if row.canteen_id == canteen_id:
            food_indices.append(indices_from_food_id[new_and_specials.loc[index].food_id])
            
    return get_recomms_df(set(food_indices), df1, columns, comment)

# utility function to get the home canteen given a user id
def get_user_home_canteen(users, user_id):
    for index, row in users[['user_id']].iterrows():
        if row.user_id == user_id:
            return users.loc[index].home_canteen
    return -1

# fetch items from previously calculated top_rated_items list
def get_top_rated_items(top_rated_items, df1, columns, comment="top rated items across canteens"):
    food_indices = []
    
    for index, row in top_rated_items.iterrows():
        food_indices.append(indices_from_food_id[top_rated_items.loc[index].food_id])
        
    return get_recomms_df(food_indices, df1, columns, comment)

# fetch items from previously calculated pop_items list
def get_popular_items(pop_items, df1, columns, comment="most popular items across canteens"):
    food_indices = []
    
    for index, row in pop_items.iterrows():
        food_indices.append(indices_from_food_id[pop_items.loc[index].food_id])
        
    return get_recomms_df(food_indices, df1, columns, comment)
    
```

### After all the hard work, we finally get the recommendations


```python
orders = pd.read_csv('./db/orders.csv')
new_and_specials = pd.read_csv('./db/new_and_specials.csv')
users = pd.read_csv('./db/users.csv')

columns = ['title', 'canteen_id', 'price', 'comment']
current_user = 2
current_canteen = get_user_home_canteen(users, current_user)


personalised_recomms(orders, df1, current_user, columns)
get_new_and_specials_recomms(new_and_specials, users, df1, current_canteen, columns)
get_top_rated_items(top_rated_items, df1, columns)
get_popular_items(pop_items, df1, columns).head(3)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>canteen_id</th>
      <th>price</th>
      <th>comment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Veg Maggi</td>
      <td>1</td>
      <td>30</td>
      <td>based on your past orders</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paneer Tikka</td>
      <td>1</td>
      <td>60</td>
      <td>based on your past orders</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Chicken Tikka</td>
      <td>1</td>
      <td>80</td>
      <td>based on your past orders</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>canteen_id</th>
      <th>price</th>
      <th>comment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cheese Maggi</td>
      <td>1</td>
      <td>25</td>
      <td>new/today's special item  in your home canteen</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>canteen_id</th>
      <th>price</th>
      <th>comment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Paneer Tikka</td>
      <td>1</td>
      <td>60</td>
      <td>top rated items across canteens</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chicken Tikka</td>
      <td>1</td>
      <td>80</td>
      <td>top rated items across canteens</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cheese Maggi</td>
      <td>1</td>
      <td>25</td>
      <td>top rated items across canteens</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>canteen_id</th>
      <th>price</th>
      <th>comment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Paneer Tikka</td>
      <td>1</td>
      <td>60</td>
      <td>most popular items across canteens</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cheese Maggi</td>
      <td>1</td>
      <td>25</td>
      <td>most popular items across canteens</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Chicken Tikka</td>
      <td>1</td>
      <td>80</td>
      <td>most popular items across canteens</td>
    </tr>
  </tbody>
</table>
</div>



These are just simple algorithms to make personalised and even general recommendations to users. We can easily use collaborative filtering or incorporate neural networks to make our prediction even better. However, these are more computationally intensive methods. Kinda overkill, IMO! Let's build that app first!
