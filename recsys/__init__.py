"""recsys — standalone acne-product recommendation system.

Couples to the rest of the repo through three file contracts only:
reads analysis.json (schema 3) + profile.json, writes recommendations.json.
No imports from src/. Stdlib only outside recsys/tools/.
"""

ENGINE_VERSION = "0.1.0"
