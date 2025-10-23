import os
from qdrant_client import QdrantClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_hybrid_search():
    """
    Connects to Qdrant and verifies if the collection is set up for hybrid search
    by checking for a full-text index on the 'content' field.
    """
    # --- Configuration ---
    # Reads connection details from environment variables, matching your app's setup.
    # You can set these in a .env file (if you use python-dotenv) or export them.
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name = os.getenv("QDRANT_COLLECTION", "documents")
    # --- End Configuration ---

    try:
        # 1. Initialize the Qdrant client
        client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=10)
        logger.info(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}...")

        # Check if the server is reachable
        client.get_collections() # This will raise an exception if connection fails
        logger.info("Successfully connected to Qdrant.")

        # 2. Get the collection information
        logger.info(f"Fetching details for collection: '{collection_name}'")
        collection_info = client.get_collection(collection_name=collection_name)

        # 3. Check for the full-text index on the 'content' field
        payload_schema = collection_info.payload_schema
        content_field_info = payload_schema.get("content")

        logger.info("\n--- Verification Results ---")
        # The schema for a text index is stored under 'params' -> 'text_index_params'
        if content_field_info and hasattr(content_field_info, 'params') and hasattr(content_field_info.params, 'text_index_params'):
            print("✅ Success! A full-text index exists on the 'content' field.")
            print("   This collection IS configured for hybrid search.")
            print("\n   Index details:")
            print(f"   {content_field_info.params.text_index_params}")
        else:
            print("❌ Not found. No full-text index on the 'content' field.")
            print("   This collection is NOT configured for hybrid search.")
            print("\n   Full payload schema:")
            # Pretty print the schema for better readability
            for field, schema in payload_schema.items():
                print(f"   - {field}: {schema}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        print(f"\n❌ Verification failed. Could not get collection info.")
        print(f"   Please check the following:")
        print(f"   1. Is the Qdrant service running and accessible at {qdrant_host}:{qdrant_port}?")
        print(f"   2. Does the collection '{collection_name}' exist?")
        print(f"   3. Are the environment variables (QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION) set correctly?")

if __name__ == "__main__":
    verify_hybrid_search()
