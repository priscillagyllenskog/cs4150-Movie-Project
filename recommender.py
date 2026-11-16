"""
Movie Recommendation System - ML class project
Phase 1 (Week 3): rule-based genre-matching recommender (pure Python)
Phase 2 (Week 4+): content-based similarity (TF-IDF + cosine) over genres,
                   keywords, cast, studio, with a rating tie-breaker.

Usage:
  python recommender.py "The Lion King" "Frozen"
  python recommender.py --method tfidf "The Lion King" "Frozen"
  python recommender.py --method both "La La Land"
  python recommender.py                      # interactive mode
Data: uses tmdb_movies.csv (from fetch_tmdb.py) automatically if it exists,
otherwise sample_movies.csv. Override with --data FILE.csv, or use the Kaggle
TMDB 5000 files via --tmdb movies.csv credits.csv (see README).
"""
import argparse, csv, math, ast, os, difflib
from collections import Counter

# ---------------------------------------------------------------- loading
def load_csv(path="sample_movies.csv"):
    """Reads sample_movies.csv or the tmdb_movies.csv written by fetch_tmdb.py."""
    movies = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not (r.get("title") or "").strip():
                continue
            try:
                rating = float(r["rating"]) if r.get("rating") else None
            except ValueError:
                rating = None
            movies.append({
                "title": r["title"].strip(),
                "genres": [g for g in (r.get("genres") or "").split("|") if g],
                "keywords": [k for k in (r.get("keywords") or "").split("|") if k],
                "cast": [c for c in (r.get("cast") or "").split("|") if c],
                "studio": (r.get("studio") or "").strip(),
                "rating": rating,
            })
    return clean(movies)

load_sample = load_csv   # old name, kept so test_edge_cases.py still works

def _names(s, limit=None):
    try:
        items = [d["name"] for d in ast.literal_eval(s)]
    except (ValueError, SyntaxError):
        items = []
    return items[:limit] if limit else items

def load_tmdb(movies_csv, credits_csv):
    """TMDB 5000 dataset (Kaggle: 'TMDB 5000 Movie Dataset')."""
    cast_by_id = {}
    with open(credits_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cast_by_id[r["movie_id"]] = _names(r["cast"], 5)
    movies = []
    with open(movies_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            studios = _names(r["production_companies"], 1)
            movies.append({
                "title": r["title"].strip(),
                "genres": _names(r["genres"]),
                "keywords": _names(r["keywords"]),
                "cast": cast_by_id.get(r["id"], []),
                "studio": studios[0] if studios else "",
                "rating": float(r["vote_average"]) if r["vote_average"] else None,
            })
    return clean(movies)

def clean(movies):
    """Week 2: remove duplicate titles, keep movies with at least one genre."""
    seen, out = set(), []
    for m in movies:
        key = m["title"].lower()
        if key in seen or not m["genres"]:
            continue
        seen.add(key)
        out.append(m)
    return out

def find(movies, title):
    """Exact (case-insensitive) match, else a unique partial match, else None."""
    t = title.strip().lower()
    for m in movies:
        if m["title"].lower() == t:
            return m
    hits = [m for m in movies if t in m["title"].lower()]
    return hits[0] if len(hits) == 1 else None

def suggest(movies, title, n=3):
    """Closest titles, for 'did you mean' messages."""
    t = title.strip().lower()
    partial = [m["title"] for m in movies if t in m["title"].lower()]
    if partial:
        return partial[:n]
    return difflib.get_close_matches(title, [m["title"] for m in movies], n=n, cutoff=0.6)

# ------------------------------------------------- phase 1: genre matching
def recommend_genre(movies, liked, k=5):
    """Score = share of the user's genre profile a candidate matches
    (genre counts across liked movies), plus a small rating bonus."""
    profile = Counter(g for m in liked for g in m["genres"])
    total = sum(profile.values()) or 1
    liked_titles = {m["title"] for m in liked}
    results = []
    for m in movies:
        if m["title"] in liked_titles:
            continue
        overlap = [g for g in m["genres"] if g in profile]
        if not overlap:
            continue
        score = sum(profile[g] for g in overlap) / total
        score += 0.02 * (m["rating"] or 0) / 10
        results.append((score, m, "shared genres: " + ", ".join(overlap)))
    results.sort(key=lambda x: -x[0])
    return results[:k]

# ------------------------------------- phase 2: TF-IDF + cosine similarity
def doc_tokens(m):
    toks = [f"g:{g}" for g in m["genres"]] * 2          # genres weighted x2
    toks += [f"k:{k}" for k in m["keywords"]]
    toks += [f"a:{a}" for a in m["cast"]]
    if m["studio"]:
        toks.append(f"s:{m['studio']}")
    return toks

def build_tfidf(movies):
    docs = [Counter(doc_tokens(m)) for m in movies]
    n = len(docs)
    df = Counter(t for d in docs for t in d)
    idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
    vecs = [{t: c * idf[t] for t, c in d.items()} for d in docs]
    return vecs

def cosine(a, b):
    dot = sum(v * b.get(t, 0) for t, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0

def recommend_tfidf(movies, liked, k=5, rating_weight=0.1):
    vecs = build_tfidf(movies)
    idx = {m["title"]: i for i, m in enumerate(movies)}
    # user preference model = average of liked movies' vectors
    pref = Counter()
    for m in liked:
        for t, v in vecs[idx[m["title"]]].items():
            pref[t] += v / len(liked)
    liked_titles = {m["title"] for m in liked}
    results = []
    for i, m in enumerate(movies):
        if m["title"] in liked_titles:
            continue
        sim = cosine(pref, vecs[i])
        if sim <= 0:
            continue
        score = (1 - rating_weight) * sim + rating_weight * (m["rating"] or 0) / 10
        shared = sorted(set(vecs[i]) & set(pref), key=lambda t: -pref[t] * vecs[i][t])[:4]
        why = "shared: " + ", ".join(t.split(":", 1)[1] for t in shared)
        results.append((score, m, why))
    results.sort(key=lambda x: -x[0])
    return results[:k]

# ------------------------------------------------------------------- CLI
def show(label, results):
    print(f"\n=== {label} ===")
    if not results:
        print("No recommendations found.")
    for rank, (score, m, why) in enumerate(results, 1):
        print(f"{rank}. {m['title']}  (score {score:.3f}, rating {m['rating']})")
        print(f"   {why}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("titles", nargs="*", help="liked movie titles")
    ap.add_argument("--method", choices=["genre", "tfidf", "both"], default="both")
    ap.add_argument("--data", default=None,
                    help="CSV to use (default: tmdb_movies.csv if it exists, else sample_movies.csv)")
    ap.add_argument("--tmdb", nargs=2, metavar=("MOVIES_CSV", "CREDITS_CSV"))
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()

    if a.tmdb:
        movies, source = load_tmdb(*a.tmdb), "TMDB 5000 files"
    else:
        path = a.data or ("tmdb_movies.csv" if os.path.exists("tmdb_movies.csv")
                          else "sample_movies.csv")
        movies, source = load_csv(path), path
    print(f"Loaded {len(movies)} movies from {source}")
    titles = a.titles or [t.strip() for t in
                          input("Liked movies (comma-separated): ").split(",") if t.strip()]
    liked, unknown = [], []
    for t in titles:
        m = find(movies, t)
        if m and m not in liked:       # repeated titles counted once
            liked.append(m)
        elif not m:
            unknown.append(t)
    for t in unknown:
        tips = suggest(movies, t)
        print(f'Not found: "{t}"' + (f" -- did you mean: {', '.join(tips)}?" if tips else ""))
    if not liked:
        print("None of the movies were found; try different titles.")
        return
    print("Based on:", ", ".join(m["title"] for m in liked))
    if a.method in ("genre", "both"):
        show("Genre matching (baseline)", recommend_genre(movies, liked, a.k))
    if a.method in ("tfidf", "both"):
        show("TF-IDF cosine similarity", recommend_tfidf(movies, liked, a.k))

if __name__ == "__main__":
    main()
