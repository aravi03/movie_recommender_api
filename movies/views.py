from django.http import HttpResponse
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD
import json
from django.http import JsonResponse

# import warnings; warnings.simpleafilter('ignore')

def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
#     for i in range(30):
#         print(sim_scores[i][1])
    return titles.iloc[movie_indices]






def abc(id):

    # md = pd.read_csv('movies_metadata.csv')
    # movie_name=md.loc[md['imdb_id'] == id]['title']
    #
    # md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(
    #     lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    # vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
    # vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
    # C = vote_averages.mean()
    # m = vote_counts.quantile(0.95)
    # md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(
    #     lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    # links_small = pd.read_csv('links_small.csv')
    # links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
    # md = md.drop([19730, 29503, 35587])
    # # Check EDA Notebook for how and why I got these indices.
    # md['id'] = md['id'].astype('int')
    # smd = md[md['id'].isin(links_small)]
    # credits = pd.read_csv('credits.csv')
    # keywords = pd.read_csv('keywords.csv')
    # keywords['id'] = keywords['id'].astype('int')
    # credits['id'] = credits['id'].astype('int')
    # md['id'] = md['id'].astype('int')
    # md = md.merge(credits, on='id')
    # md = md.merge(keywords, on='id')
    # smd = md[md['id'].isin(links_small)]
    # smd['cast'] = smd['cast'].apply(literal_eval)
    # smd['crew'] = smd['crew'].apply(literal_eval)
    # smd['keywords'] = smd['keywords'].apply(literal_eval)
    # smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
    # smd['crew_size'] = smd['crew'].apply(lambda x: len(x))
    #
    # def get_director(x):
    #     for i in x:
    #         if i['job'] == 'Director':
    #             return i['name']
    #     return np.nan
    # smd['director'] = smd['crew'].apply(get_director)
    # smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    # smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >= 3 else x)
    # smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    # smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
    # smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
    # smd['director'] = smd['director'].apply(lambda x: [x, x, x])
    # s = smd.apply(lambda x: pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
    # s.name = 'keyword'
    # s = s.value_counts()
    # s[:5]
    # s = s[s > 1]
    # stemmer = SnowballStemmer('english')
    # stemmer.stem('dogs')
    #
    # def filter_keywords(x):
    #     words = []
    #     for i in x:
    #         if i in s:
    #             words.append(i)
    #     return words
    #
    # smd['keywords'] = smd['keywords'].apply(filter_keywords)
    # smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
    # smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
    # smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
    # smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))
    # smd.to_csv('mysmd.csv')
    # newmd=smd[['title', 'vote_count', 'vote_average', 'year', 'imdb_id', 'soup']]
    # newmd.to_csv('newsmd.csv')

    smd = pd.read_csv('newsmd.csv')
    movie_name = smd.loc[smd['imdb_id'] == id]['title']

    count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    count_matrix = count.fit_transform(smd['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    smd = smd.reset_index()
    titles = smd[['title', 'imdb_id']]
    indices = pd.Series(smd.index, index=smd['title'])

    def improved_recommendations(title):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:26]
        movie_indices = [i[0] for i in sim_scores]

        movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'imdb_id']]
        vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
        vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
        C = vote_averages.mean()
        m = vote_counts.quantile(0.60)
        qualified = movies[
            (movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
        qualified['vote_count'] = qualified['vote_count'].astype('int')
        qualified['vote_average'] = qualified['vote_average'].astype('int')

        def weighted_rating(x):
            v = x['vote_count']
            R = x['vote_average']
            return (v / (v + m) * R) + (m / (m + v) * C)
        qualified['wr'] = qualified.apply(weighted_rating, axis=1)
        qualified = qualified.sort_values('wr', ascending=False).head(10)
        return qualified
    jsonn = improved_recommendations(movie_name.values[0])[['imdb_id', 'wr', 'title']].values
    print(movie_name.values[0])
    return jsonn

def index(request):
    id=request.GET["id"]
    jsonn=abc(id)
    print(request.GET["id"])
    # json_numbers = json.dumps(jsonn)
    # from django.http import JsonResponse
    #

    val = []
    # print(jsonn)
    for i in jsonn:
        data = {
            "imdb_id": i[0],
            "score": i[1],
            "title": i[2]
        }
        json_data = json.dumps(data)
        val.append(json_data)

    return JsonResponse({"recom":val}, safe=False)

def home(request):
    return HttpResponse("Welcome to Home page")