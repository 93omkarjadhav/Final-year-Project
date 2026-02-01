# # redis_config.py
# import os
# from upstash_redis import Redis
# from dotenv import load_dotenv
# from langchain_community.cache import UpstashRedisCache
# from langchain.globals import set_llm_cache

# def init_redis_cache():
#     """Connects to Upstash Redis and sets up the global LangChain cache."""
#     load_dotenv()
    
#     url = os.getenv("UPSTASH_REDIS_REST_URL")
#     token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
    
#     if not url or not token:
#         print("⚠️ Redis credentials not found in .env")
#         return None

#     try:
#         # Initialize the Upstash Redis client
#         redis_client = Redis(url=url, token=token)
        
#         # Wrap it for LangChain
#         cache = UpstashRedisCache(redis_client=redis_client)
        
#         # Apply the cache globally so every LLM call uses it
#         set_llm_cache(cache)
        
#         print("✅ Upstash Redis Caching is now ACTIVE")
#         return redis_client
#     except Exception as e:
#         print(f"❌ Redis Connection Error: {e}")
#         return None

#222222
# redis_config.py - UPDATED
# import os
# from upstash_redis import Redis
# from dotenv import load_dotenv
# from langchain_community.cache import UpstashRedisCache
# from langchain_core.globals import set_llm_cache  # FIX: Changed from langchain.globals
# #from langchain.globals import set_llm_cache

# def init_redis_cache():
#     load_dotenv()
#     url = os.getenv("UPSTASH_REDIS_REST_URL")
#     token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
    
#     if not url or not token:
#         return None

#     try:
#         # Initialize the Upstash Redis client
#         redis_client = Redis(url=url, token=token)
#         # Test the connection first
#         redis_client.ping()
#         print("✅ Redis connection successful")
#         # FIX: Some versions of LangChain expect 'redis_' or just the positional client
#         # Try this standard way for the latest langchain-community:
#         cache = UpstashRedisCache(redis_=redis_client) 
        
#         # Apply the cache globally
#         set_llm_cache(cache)
#         return redis_client
#     except Exception as e:
#         # This will print the exact error in your terminal to help us debug
#         print(f"❌ Redis Connection Error: {e}")
#         return None

#3333333
# redis_config.py - CORRECTED VERSION
import os
from upstash_redis import Redis
from dotenv import load_dotenv
from langchain_community.cache import UpstashRedisCache
from langchain_core.globals import set_llm_cache  # FIX: Changed from langchain.globals

def init_redis_cache():
    load_dotenv()
    url = os.getenv("UPSTASH_REDIS_REST_URL")
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
    
    if not url or not token:
        print("⚠️ Redis credentials not found in .env")
        return None

    try:
        # Initialize the Upstash Redis client
        redis_client = Redis(url=url, token=token)
        
        # Test the connection first
        redis_client.ping()
        print("✅ Redis connection successful")
        
        # FIX: Try different parameter names based on LangChain version
        # Most common is 'redis_client' or positional argument
        try:
            cache = UpstashRedisCache(redis_client=redis_client)
        except TypeError:
            # Fallback: try positional argument
            try:
                cache = UpstashRedisCache(redis_client)
            except TypeError:
                # Another fallback: try 'redis_' parameter
                cache = UpstashRedisCache(redis_=redis_client)
        
        # Apply the cache globally
        set_llm_cache(cache)
        print("✅ Upstash Redis Caching is now ACTIVE")
        
        # Test write to verify it works
        redis_client.set("langchain_cache_test", "active", ex=60)
        print("✅ Cache write test successful")
        
        return redis_client
    except Exception as e:
        print(f"❌ Redis Connection Error: {e}")
        import traceback
        traceback.print_exc()
        return None