# Food-Recmmendation-System-Python


This is the Recommendation Engine that will be used in building the <b>Lunchbox App</b>, a platform for ordering food and keeping track of user expenditure and canteen sales. Regardless of whether or not this is actually implemented in all the canteens of <b>IIT Kanpur</b> (given the potential for frauds & cyber-attacks) I will still complete the platform.

Also, <b>I would be open-sourcing the app</b> so that any campus can implement a <b>cash-less & integrated system of ordering food</b> across their whole campus. After all, what good are IITs for if our canteens still keep track of student accounts on paper registers!

## Build instructions

- ```git clone https://github.com/gsunit/Food-Recommendation-System-Pyhton.git```
- Run the Jupyter Notebook `src.ipynb`



## Demographic Filtering

Suggesting the items that are well-received and popular among the users. Most trending items and items with the best rating rise to the top and get shortlisted for recommendation.


```python
import pandas as pd 
import numpy as np

# Importing db of food items across all canteens registered on the platform
df1=pd.read_csv('./db/food.csv')
df1.columns = ['food_id','title','canteen_id','price', 'num_orders', 'category', 'avg_rating', 'num_rating', 'tags']

df1
```




<div>

<table border="1" class="dataframe" style="height:3px;">
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


##  Results of demographic filtering
```python
top_rated_items[['title', 'num_rating', 'avg_rating', 'score']].head()
pop_items[['title', 'num_orders']].head()
```




<div>
<table class="dataframe">
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

A bit more personalised recommendation. We will analyse the past orders of the user and suggest back those items which are similar.

Also, since each person has a "home canteen", the user should be notified of any new items included in the menu by the vendor.

We will be use <b>Count Vectorizer</b> from <b>Scikit-Learn</b> to find similarity between items based on their title, category and tags. To bring all these properties of each item together, we create a <b>"soup"</b> of tags. <b>"Soup"</b> is a processed string correspnding to each item, formed using the constituents of tags, tile and category.

<div>
<table class="dataframe">
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


## Using CountVectorizer from Scikit-Learn

```python
# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(stop_words='english')

# df1['soup']
count_matrix = count.fit_transform(df1['soup'])

# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(count_matrix, count_matrix)
```


## Sample Recommendation 

```python
df1.loc[get_recommendations(title="Paneer Tikka")]
```




<div>
<table class="dataframe">
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





### After all the hard work, we finally get the recommendations


```python
personalised_recomms(orders, df1, current_user, columns)
get_new_and_specials_recomms(new_and_specials, users, df1, current_canteen, columns)
get_top_rated_items(top_rated_items, df1, columns)
get_popular_items(pop_items, df1, columns).head(3)
```




<div>
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



These are just simple algorithms to make personalised & general recommendations to users. We can easily use collaborative filtering or incorporate neural networks to make our prediction even better. However, these are more computationally intensive methods. Kinda overkill, IMO! Let's build that app first, then move on to other features!

#### Star the repository and send in your PRs if you think the engine needs any improvement or help me implement some more advanced features.
