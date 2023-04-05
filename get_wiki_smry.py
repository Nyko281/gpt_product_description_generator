import json
import pandas as pd
import wikipediaapi


print("starts collecting wikipedia data")

with open('') as input:                                                                                     # open file with all keywords
    data = json.load(input)

key_set = set()

for key, value in data.items():
    for item in value[2]:                                                           
        key_set.add(item)                                                                                   # put all keywords in set for uniqueness

wiki = wikipediaapi.Wikipedia('de')                                                                         # initialize wikipedia with needed language

def page_exists(p):                                                                                         # function to test if word has a page
    page = wiki.page(p)
    try:
        page.exists()
    except:
        return False
    else:
        test = page.exists()
        return test


def get_wiki_sum(w):                                                                                        # function that returns wikepedia summary for word
    page = wiki.page(w)
    try:
        page.summary
    except:
        return False
    else:
        wiki_sum = page.summary
        return wiki_sum


df = pd.DataFrame(columns=['id', 'Titel', 'Marketingtext'])                                                 # create empyt data frame with only column names

c = 1
for key in key_set:                                                                                         # iterate through all keywords
    if page_exists(key) and key != None:
        row = pd.DataFrame({"id":[c],
                            "Titel":[key],
                            "Marketingtext":[get_wiki_sum(key)]})                                           # get text for keyword
        df = pd.concat([df, row], ignore_index=True)                                                        # append id, keyword and related summary to data frame

    if c % 500 == 0 or c == 1:
        print(f"{c} / {len(key_set)} processed")
    c = c + 1

df.to_excel("", index=False)                                                                                # safe data frame
