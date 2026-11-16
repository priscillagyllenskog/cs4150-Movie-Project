## Week map
- Week 1-2: `sample_movies.csv` is a small starter set. For the real project, download the
  **TMDB 5000 Movie Dataset** (Kaggle) and run
  `python recommender.py --tmdb tmdb_5000_movies.csv tmdb_5000_credits.csv "Avatar"`.
  `clean()` removes duplicates and movies with no genres.
- Week 3: `recommend_genre` = baseline rule-based genre matching.
- Week 4: edge-case tests (`python test_edge_cases.py`), then compare against TF-IDF.
- Week 5: `recommend_tfidf` adds keywords/cast/studio + rating; compare lists for the writeup.
