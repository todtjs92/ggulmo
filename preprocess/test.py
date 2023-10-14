import pickle
with open('../data/user_click_items.pickle','rb') as f:
    a = pickle.load(f)
print(a)


