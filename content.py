import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Food data
food = pd.DataFrame({
    "Food_Name": ["Salmon", "Pizza", "Salad", "Biryani", "Burger"],
    "Cuisine": ["American", "Italian", "Italian", "Indian", "American"],
    "Diet": ["NonVeg", "Veg", "Veg", "NonVeg", "Veg"],
    "Calories": [450, 700, 350, 800, 500]
})

# Ratings
ratings = pd.DataFrame({
    "Salmon": [5, np.nan, 4, 3],
    "Pizza": [4, 5, np.nan, 3],
    "Salad": [np.nan, 4, 5, np.nan],
    "Biryani": [2, 3, 5, 4],
    "Burger": [3, np.nan, 3, 5]
}, index=["User1","User2","User3","User4"])

# New item
new = pd.DataFrame({
    "Food_Name": ["VegPizza"],
    "Cuisine": ["Italian"],
    "Diet": ["Veg"],
    "Calories": [300]
})

# Combine
all_food = pd.concat([food, new], ignore_index=True)

# Encode
enc = OneHotEncoder()
cat = enc.fit_transform(all_food[["Cuisine","Diet"]]).toarray()
cal = all_food[["Calories"]] / all_food["Calories"].max()

features = np.hstack([cat, cal])

# Similarity
sim = cosine_similarity(features)

# Similarity of new item
new_sim = sim[-1][:-1]

# Input
user = input("Enter user: ")

user_r = ratings.loc[user]

# Top 3 similar items
top = np.argsort(new_sim)[-3:][::-1]
top_items = food.iloc[top]["Food_Name"].values

# Get ratings
sim_scores = new_sim[top]
r = user_r[top_items].values

# Remove NaN
mask = ~np.isnan(r)

if mask.sum() == 0:
    print("No rating available")
else:
    pred = np.dot(sim_scores[mask], r[mask]) / sim_scores[mask].sum()
    print("Predicted rating:", round(pred,2))

    if pred >= 3:
        print("Recommend VegPizza")
    else:
        print("Do NOT recommend")
