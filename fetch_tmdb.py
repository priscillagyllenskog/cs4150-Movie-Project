"""
Download a movie catalog from the TMDB API into a CSV that recommender.py can read.

Setup (each teammate, once):
  1. Create a free TMDB account and copy your "API Read Access Token"
     (Settings -> API).
  2. Set it as an environment variable. Never paste it into code or commit it.
       macOS/Linux:  export TMDB_TOKEN="your_token"
       Windows PS:   $env:TMDB_TOKEN="your_token"
  3. pip install requests

Usage:
  python fetch_tmdb.py                         # ~200 popular movies -> tmdb_movies.csv
  python fetch_tmdb.py --pages 25              # ~500 movies
  python fetch_tmdb.py --source top_rated --out top.csv
Then:
  python recommender.py --data tmdb_movies.csv "Inception"

This product uses the TMDB API but is not endorsed or certified by TMDB.
"""
import argparse, csv, os, sys, time
import requests

BASE = "https://api.themoviedb.org/3"

def make_session():
    token = os.environ.get("TMDB_TOKEN")
    if not token:
        sys.exit("Set the TMDB_TOKEN environment variable to your TMDB Read Access Token.")
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}", "accept": "application/json"})
    return s

def get(session, path, retries=4, **params):
    """GET with retry on rate limits (429) and temporary server errors."""
    for attempt in range(retries):
        try:
            r = session.get(f"{BASE}/{path}", params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 401:
            sys.exit("401 Unauthorized: check that TMDB_TOKEN is the Read Access Token "
                     "(no extra spaces).")
        if r.status_code == 429 or r.status_code >= 500:
            wait = int(r.headers.get("Retry-After", 2 ** attempt))
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Failed after {retries} tries: {path}")

def movie_row(d):
    studios = d.get("production_companies") or []
    return {
        "title": d["title"],
        "genres": "|".join(g["name"] for g in d.get("genres", [])),
        "keywords": "|".join(k["name"] for k in d.get("keywords", {}).get("keywords", [])),
        "cast": "|".join(c["name"] for c in d.get("credits", {}).get("cast", [])[:5]),
        "studio": studios[0]["name"] if studios else "",
        "rating": d.get("vote_average", ""),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=10, help="20 movies per page")
    ap.add_argument("--source", choices=["popular", "top_rated"], default="popular")
    ap.add_argument("--out", default="tmdb_movies.csv")
    a = ap.parse_args()

    session = make_session()
    rows, seen = [], set()
    for page in range(1, a.pages + 1):
        listing = get(session, f"movie/{a.source}", page=page)
        for m in listing["results"]:
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            d = get(session, f"movie/{m['id']}", append_to_response="credits,keywords")
            rows.append(movie_row(d))
        print(f"page {page}/{a.pages}: {len(rows)} movies", flush=True)
        if page >= listing.get("total_pages", page):
            break

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["title", "genres", "keywords", "cast", "studio", "rating"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} movies to {a.out}")
    print("Credit TMDB: This product uses the TMDB API but is not endorsed or certified by TMDB.")

if __name__ == "__main__":
    main()
