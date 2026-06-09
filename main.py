from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
from hybrid import HybridRecommender

print("=" * 40)
print("   MOVIE RECOMMENDATION SYSTEM")
print("=" * 40)

print("\n📽️  Content-Based Recommendations")
print("-" * 40)
cb = ContentBasedRecommender()
cb.recommend("The Dark Knight")

print("\n👥  Collaborative Filtering")
print("-" * 40)
collab = CollaborativeRecommender()
collab.recommend("User3")

print("\n🔀  Hybrid Recommendations")
print("-" * 40)
hybrid = HybridRecommender(cb_weight=0.6, collab_weight=0.4)
hybrid.recommend("Inception", user="User2")