import faiss
import psycopg2
import numpy as np
import redis
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import atexit

load_dotenv()

# ----------------- LTM (Long-Term Memory) -----------------
embedding_dim = 1536  # Example dimension for OpenAI embeddings
vector_store = faiss.IndexFlatL2(embedding_dim)

# PostgreSQL Connection (graceful: app continues without DB if unavailable)
conn = None
cursor = None
try:
    pg_password = os.getenv("POSTGRES_PASSWORD", "")
    conn = psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB", "ltm_database"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=pg_password,
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432")
    )
    cursor = conn.cursor()
    print("Successfully connected to PostgreSQL database")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ltm (
            user_id TEXT PRIMARY KEY,
            workout_history JSONB
        )
    """)
    conn.commit()
except Exception as e:
    print(f"Warning: PostgreSQL unavailable – LTM features disabled. ({e})")
    conn = None
    cursor = None

def store_ltm(user_id, workout_data):
    """
    Store user's past workouts in Long-Term Memory (LTM).
    """
    if not user_id or not workout_data:
        return "Error: Invalid user_id or workout_data"
    if conn is None or cursor is None:
        return "Error: PostgreSQL not available"

    try:
        # Get existing history
        cursor.execute("SELECT workout_history FROM ltm WHERE user_id = %s", (user_id,))
        result = cursor.fetchone()
        
        # Create new history entry
        new_entry = {
            "timestamp": datetime.now().isoformat(),
            "data": workout_data,
            "level": workout_data.get('level', 'beginner') if isinstance(workout_data, dict) else None
        }
        
        if result:
            # Append to existing history
            history = json.loads(result[0]) if isinstance(result[0], str) else result[0]
            if not isinstance(history, list):
                history = [history]
            history.append(new_entry)
        else:
            # Create new history
            history = [new_entry]
            print(f"Creating new history for user {user_id}")

        # Store embedding
        embedding = np.random.rand(embedding_dim).astype('float32')
        vector_store.add(np.array([embedding]))

        # Update database
        cursor.execute("""
            INSERT INTO ltm (user_id, workout_history) 
            VALUES (%s, %s::jsonb) 
            ON CONFLICT (user_id) 
            DO UPDATE SET workout_history = EXCLUDED.workout_history;
        """, (user_id, json.dumps(history)))
        
        conn.commit()
        print(f"\n=== LTM Storage Debug ===")
        print(f"User ID: {user_id}")
        print(f"Entry Type: {'New' if not result else 'Update'}")
        print(f"History Length: {len(history)}")
        print("======================\n")
        
        return f"User {user_id}'s workout history updated in PostgreSQL LTM."
    except Exception as e:
        print(f"Error storing in LTM: {e}")
        return f"Error storing in LTM: {str(e)}"

def get_ltm(user_id):
    """
    Retrieve user history from Long-Term Memory (LTM) in PostgreSQL.
    """
    if not user_id:
        return "Error: Invalid user_id"
    if conn is None or cursor is None:
        return []

    try:
        cursor.execute("SELECT workout_history FROM ltm WHERE user_id = %s", (user_id,))
        result = cursor.fetchone()
        
        # If no result found, return empty list
        if not result:
            print(f"No history found for user {user_id}")
            return []
            
        # Parse the history data
        try:
            history = json.loads(result[0]) if isinstance(result[0], str) else result[0]
            
            # Ensure history is a list
            if not isinstance(history, list):
                print(f"Converting non-list history to list for user {user_id}")
                history = [history] if history else []
            
            # Debug logging
            print(f"\n=== LTM Retrieval Debug ===")
            print(f"User ID: {user_id}")
            print(f"History Type: {type(history)}")
            print(f"History Entries: {len(history)}")
            print("=========================\n")
            
            return history
            
        except json.JSONDecodeError as e:
            print(f"Error parsing history data for user {user_id}: {e}")
            return []
            
    except Exception as e:
        print(f"Database error retrieving LTM data: {e}")
        return []

# ----------------- STM (Short-Term Memory) -----------------
redis_client = None
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        decode_responses=True
    )
    redis_client.ping()
    print("Successfully connected to Redis")
except Exception as e:
    print(f"Warning: Redis unavailable – STM features disabled. ({e})")
    redis_client = None

def store_stm(user_id, metric):
    """
    Store real-time workout data in Short-Term Memory (STM) using Redis.
    """
    if not user_id or not metric:
        return "Error: Invalid user_id or metric"
    if redis_client is None:
        return "Error: Redis not available"

    try:
        # Store with timestamp
        data = {
            "timestamp": datetime.now().isoformat(),
            "feedback": metric
        }
        redis_client.set(user_id, json.dumps(data))
        return f"Real-time workout data for {user_id} stored in STM."
    except Exception as e:
        return f"Error storing in STM: {str(e)}"

def get_stm(user_id):
    """
    Retrieve real-time workout data from Short-Term Memory (STM) using Redis.
    """
    if not user_id:
        return "Error: Invalid user_id"
    if redis_client is None:
        return None

    try:
        data = redis_client.get(user_id)
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        return f"Error retrieving from STM: {str(e)}"

def clear_user_data(user_id):
    """
    Clear all data for a specific user from both LTM and STM.
    Useful for testing and user data management.
    """
    if conn is None and redis_client is None:
        return "Error: No database connections available"
    try:
        # Clear from PostgreSQL
        cursor.execute("DELETE FROM ltm WHERE user_id = %s", (user_id,))
        conn.commit()
        
        # Clear from Redis
        redis_client.delete(user_id)
        
        print(f"\n=== User Data Cleared ===")
        print(f"User ID: {user_id}")
        print(f"LTM and STM data cleared")
        print("=======================\n")
        
        return f"All data cleared for user {user_id}"
    except Exception as e:
        print(f"Error clearing user data: {e}")
        return f"Error clearing user data: {str(e)}"
    
def check_connections():
    """
    Check if database connections are healthy
    """
    try:
        pg_ok = False
        redis_ok = False
        if conn is not None and cursor is not None:
            cursor.execute("SELECT 1")
            pg_ok = True
        if redis_client is not None:
            redis_client.ping()
            redis_ok = True
        return pg_ok and redis_ok
    except Exception as e:
        print(f"Connection health check failed: {e}")
        return False

def get_user_stats(user_id):
    """
    Get statistics about user's stored data
    """
    try:
        ltm_history = get_ltm(user_id)
        stm_data = get_stm(user_id)
        
        stats = {
            "ltm_entries": len(ltm_history),
            "has_stm_data": stm_data is not None,
            "last_activity": ltm_history[-1]["timestamp"] if ltm_history else None,
            "user_level": ltm_history[-1].get("level", "beginner") if ltm_history else "beginner"
        }
        
        return stats
    except Exception as e:
        print(f"Error getting user stats: {e}")
        return None    

def get_memory_usage():
    """
    Get current memory usage statistics
    """
    try:
        # Get PostgreSQL table size
        cursor.execute("""
            SELECT pg_size_pretty(pg_total_relation_size('ltm'))
        """)
        pg_size = cursor.fetchone()[0]
        
        # Get Redis memory info
        redis_info = redis_client.info(section='memory')
        redis_used = redis_info.get('used_memory_human', 'N/A')
        
        stats = {
            "postgresql_size": pg_size,
            "redis_memory": redis_used,
            "vector_store_size": vector_store.ntotal
        }
        
        return stats
    except Exception as e:
        print(f"Error getting memory usage: {e}")
        return None

def cleanup_connections():
    """
    Cleanup database and Redis connections
    """
    try:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()
        if redis_client is not None:
            redis_client.close()
        print("Database connections closed successfully")
    except Exception as e:
        print(f"Error closing connections: {e}")

# Register cleanup function
atexit.register(cleanup_connections)