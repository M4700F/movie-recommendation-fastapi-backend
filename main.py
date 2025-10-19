# main.py - FastAPI Server for Movie Recommendations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import tensorflow as tf
import pickle
import joblib
from collections import defaultdict

app = FastAPI(title="Movie Recommendation API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and data
model = None
user_NN = None
item_NN = None
scalerUser = None
scalerItem = None
scaler_y = None
item_vecs = None
movie_feature_vectors = None
movie_dict = None
config = None


@app.on_event("startup")
async def load_models():
    """Load all models and artifacts on startup"""
    global model, user_NN, item_NN, scalerUser, scalerItem, scaler_y
    global item_vecs, movie_feature_vectors, movie_dict, config
    
    print("Loading models...")
    
    try:
        # Load TensorFlow models
        model = tf.keras.models.load_model('./models/recommendation_model.h5')
        user_NN = tf.keras.models.load_model('./models/user_nn.h5')
        item_NN = tf.keras.models.load_model('./models/item_nn.h5')
        print("✓ Neural network models loaded")
        
        # Load scalers
        scalerUser = joblib.load('./models/scaler_user.pkl')
        scalerItem = joblib.load('./models/scaler_item.pkl')
        scaler_y = joblib.load('./models/scaler_y.pkl')
        print("✓ Scalers loaded")
        
        # Load configuration
        with open('./models/config.pkl', 'rb') as f:
            config = pickle.load(f)
        print("✓ Configuration loaded")
        
        # Load item vectors and movie features
        item_vecs = np.load('./models/item_vecs.npy')
        movie_feature_vectors = np.load('./models/movie_feature_vectors.npy')
        print("✓ Item vectors loaded")
        
        # Load movie dictionary
        with open('./models/movie_dict.pkl', 'rb') as f:
            movie_dict = pickle.load(f)
        print("✓ Movie dictionary loaded")
        
        print(f"\n📊 Loaded {len(movie_dict)} movies")
        print(f"📊 Item vectors shape: {item_vecs.shape}")
        print(f"📊 Movie feature vectors shape: {movie_feature_vectors.shape}")
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        raise e


# Request/Response Models
class UserRating(BaseModel):
    movie_id: int
    rating: float  # 0.5 to 5.0


class NewUserPreferences(BaseModel):
    action: float = 1.0
    adventure: float = 1.0
    animation: float = 1.0
    childrens: float = 1.0
    comedy: float = 1.0
    crime: float = 1.0
    documentary: float = 1.0
    drama: float = 1.0
    fantasy: float = 1.0
    horror: float = 1.0
    mystery: float = 1.0
    romance: float = 1.0
    scifi: float = 1.0
    thriller: float = 1.0


class RecommendationRequest(BaseModel):
    user_id: int
    user_ratings: Optional[List[UserRating]] = None
    new_user_preferences: Optional[NewUserPreferences] = None
    top_n: int = 10


class MovieRecommendation(BaseModel):
    movie_id: int
    predicted_score: float
    title: str
    genres: str


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[MovieRecommendation]


def build_user_vector_from_ratings(user_ratings: List[UserRating], user_id: int):
    """
    Build user vector from existing ratings
    Similar to Andrew Ng's approach - create genre preferences from rated movies
    """
    # Initialize user vector
    # Format: [user_id, rating_count, rating_avg, genre_preferences...]
    genre_count = 14  # Number of genres
    genre_ratings = defaultdict(list)
    
    total_rating = 0.0
    rating_count = len(user_ratings)
    
    for rating in user_ratings:
        movie_id = rating.movie_id
        if movie_id not in movie_dict:
            continue
        
        # Get movie genres
        genres = movie_dict[movie_id]['genres'].split('|')
        
        # Add rating to each genre
        for genre in genres:
            genre_ratings[genre.strip()].append(rating.rating)
        
        total_rating += rating.rating
    
    # Calculate average rating
    rating_avg = total_rating / rating_count if rating_count > 0 else 0.0
    
    # Calculate genre averages (in order from user_features)
    genre_order = config['user_features'][3:]  # Skip user_id, rating_count, rating_avg
    genre_avgs = []
    
    for genre in genre_order:
        if genre in genre_ratings:
            genre_avgs.append(np.mean(genre_ratings[genre]))
        else:
            genre_avgs.append(1.0)  # Default neutral rating
    
    # Build complete user vector
    user_vec = np.array([[user_id, rating_count, rating_avg] + genre_avgs])
    
    return user_vec


def build_user_vector_from_preferences(preferences: NewUserPreferences, user_id: int):
    """
    Build user vector from explicit genre preferences (for new users)
    """
    rating_count = 1  # Minimal count for new users
    rating_avg = np.mean([
        preferences.action, preferences.adventure, preferences.animation,
        preferences.childrens, preferences.comedy, preferences.crime,
        preferences.documentary, preferences.drama, preferences.fantasy,
        preferences.horror, preferences.mystery, preferences.romance,
        preferences.scifi, preferences.thriller
    ])
    
    user_vec = np.array([[
        user_id, rating_count, rating_avg,
        preferences.action, preferences.adventure, preferences.animation,
        preferences.childrens, preferences.comedy, preferences.crime,
        preferences.documentary, preferences.drama, preferences.fantasy,
        preferences.horror, preferences.mystery, preferences.romance,
        preferences.scifi, preferences.thriller
    ]])
    
    return user_vec


def predict_for_user(user_vec: np.ndarray, top_n: int = 10, 
                     rated_movie_ids: set = None):
    """
    Predict ratings for all movies for a given user vector
    """
    if rated_movie_ids is None:
        rated_movie_ids = set()
    
    # Replicate user vector to match all movies
    num_items = len(item_vecs)
    user_vecs = np.tile(user_vec, (num_items, 1))
    
    # Scale the vectors
    u_s = config['u_s']
    i_s = config['i_s']
    
    scaled_user_vecs = scalerUser.transform(user_vecs)
    scaled_item_vecs = scalerItem.transform(item_vecs)
    
    # Make predictions
    y_p = model.predict(
        [scaled_user_vecs[:, u_s:], scaled_item_vecs[:, i_s:]],
        verbose=0
    )
    
    # Inverse transform predictions
    y_pu = scaler_y.inverse_transform(y_p)
    
    # Create recommendations list
    recommendations = []
    for i, movie_id in enumerate(item_vecs[:, 0]):
        movie_id = int(movie_id)
        
        # Skip already rated movies
        if movie_id in rated_movie_ids:
            continue
        
        if movie_id not in movie_dict:
            continue
        
        recommendations.append({
            'movie_id': movie_id,
            'predicted_score': float(y_pu[i, 0]),
            'title': movie_dict[movie_id]['title'],
            'genres': movie_dict[movie_id]['genres']
        })
    
    # Sort by predicted score and return top N
    recommendations.sort(key=lambda x: x['predicted_score'], reverse=True)
    
    return recommendations[:top_n]


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get movie recommendations for a user
    
    Two modes:
    1. Existing user with ratings: Use ratings to build profile
    2. New user with preferences: Use explicit genre preferences
    """
    try:
        rated_movie_ids = set()
        
        # Build user vector based on input type
        if request.user_ratings and len(request.user_ratings) > 0:
            # Existing user with ratings
            user_vec = build_user_vector_from_ratings(
                request.user_ratings, 
                request.user_id
            )
            rated_movie_ids = {r.movie_id for r in request.user_ratings}
            
        elif request.new_user_preferences:
            # New user with explicit preferences
            user_vec = build_user_vector_from_preferences(
                request.new_user_preferences,
                request.user_id
            )
            
        else:
            raise HTTPException(
                status_code=400, 
                detail="Either user_ratings or new_user_preferences must be provided"
            )
        
        # Get predictions
        recommendations = predict_for_user(
            user_vec, 
            request.top_n,
            rated_movie_ids
        )
        
        # Convert to response format
        rec_list = [
            MovieRecommendation(
                movie_id=rec['movie_id'],
                predicted_score=rec['predicted_score'],
                title=rec['title'],
                genres=rec['genres']
            )
            for rec in recommendations
        ]
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=rec_list
        )
        
    except Exception as e:
        print(f"Error in recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/similar-movies/{movie_id}")
async def get_similar_movies(movie_id: int, top_n: int = 10):
    """
    Find movies similar to a given movie using feature vectors
    """
    try:
        # Find the movie in item_vecs
        movie_indices = np.where(item_vecs[:, 0] == movie_id)[0]
        
        if len(movie_indices) == 0:
            raise HTTPException(status_code=404, detail="Movie not found")
        
        # Get the movie's feature vector (use first occurrence)
        movie_idx = movie_indices[0]
        movie_vec = movie_feature_vectors[movie_idx]
        
        # Calculate cosine similarity with all movies
        similarities = np.dot(movie_feature_vectors, movie_vec)
        
        # Get top similar movies (excluding the movie itself)
        similar_indices = np.argsort(-similarities)
        
        similar_movies = []
        for idx in similar_indices:
            mid = int(item_vecs[idx, 0])
            if mid == movie_id:
                continue
            
            if mid not in movie_dict:
                continue
            
            similar_movies.append({
                'movie_id': mid,
                'similarity_score': float(similarities[idx]),
                'title': movie_dict[mid]['title'],
                'genres': movie_dict[mid]['genres']
            })
            
            if len(similar_movies) >= top_n:
                break
        
        return {'similar_movies': similar_movies}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": model is not None,
        "num_movies": len(movie_dict) if movie_dict else 0,
        "num_item_vectors": len(item_vecs) if item_vecs is not None else 0
    }


@app.get("/genres")
async def get_genres():
    """Get list of all genres"""
    if config is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    genres = config['user_features'][3:]  # Skip user_id, rating_count, rating_avg
    return {"genres": genres}


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)