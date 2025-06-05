import os
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
except ImportError as e:
    raise ImportError(
        "Qdrant client not installed. Please install it with: pip install qdrant-client"
    ) from e

load_dotenv()


class QdrantRAGClient:
    """Qdrant client wrapper for direct Qdrant operations."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
    ):
        """
        Initialize Qdrant client.

        Args:
            host: Qdrant host (default from QDRANT_HOST env var or localhost)
            port: Qdrant port (default from QDRANT_PORT env var or 6333)
            url: Complete Qdrant URL (overrides host/port)
            api_key: API key for Qdrant cloud (from QDRANT_API_KEY env var)
            timeout: Request timeout in seconds
        """
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = int(port or os.getenv("QDRANT_PORT", "6333"))
        self.url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.timeout = timeout

        # Initialize client
        if self.url:
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=self.timeout
            )
        else:
            self.client = QdrantClient(
                host=self.host,
                port=self.port,
                timeout=self.timeout
            )

        # Validate connection
        self._validate_connection()

    def _validate_connection(self) -> bool:
        """Validate connection to Qdrant server."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Qdrant server: {str(e)}")

    def health_check(self) -> bool:
        """Check if Qdrant server is healthy."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        try:
            collections = self.client.get_collections()
            return any(col.name == collection_name for col in collections.collections)
        except Exception:
            return False

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
        recreate: bool = False
    ) -> bool:
        """
        Create a collection in Qdrant.

        Args:
            collection_name: Name of the collection
            vector_size: Size of the vectors to store
            distance: Distance metric (COSINE, EUCLID, DOT)
            recreate: Whether to recreate collection if it exists

        Returns:
            True if collection was created or already exists
        """
        try:
            if recreate and self.collection_exists(collection_name):
                self.client.delete_collection(collection_name)

            if not self.collection_exists(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=distance
                    )
                )
            return True
        except Exception as e:
            raise Exception(f"Failed to create collection {collection_name}: {str(e)}")

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            if self.collection_exists(collection_name):
                self.client.delete_collection(collection_name)
            return True
        except Exception as e:
            raise Exception(f"Failed to delete collection {collection_name}: {str(e)}")

    def upsert_points(
        self,
        collection_name: str,
        points: List[Dict[str, Any]]
    ) -> bool:
        """
        Upsert points into a collection.

        Args:
            collection_name: Name of the collection
            points: List of points with 'id', 'vector', and 'payload' keys

        Returns:
            True if successful
        """
        try:
            qdrant_points = []
            for point in points:
                qdrant_points.append(
                    PointStruct(
                        id=point.get("id", str(uuid.uuid4())),
                        vector=point["vector"],
                        payload=point.get("payload", {})
                    )
                )

            self.client.upsert(
                collection_name=collection_name,
                points=qdrant_points
            )
            return True
        except Exception as e:
            raise Exception(f"Failed to upsert points to {collection_name}: {str(e)}")

    def search_points(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar points in a collection.

        Args:
            collection_name: Name of the collection
            query_vector: Query vector for similarity search
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filter_conditions: Additional filter conditions

        Returns:
            List of search results with 'id', 'score', and 'payload'
        """
        try:
            search_params = {
                "collection_name": collection_name,
                "query_vector": query_vector,
                "limit": limit
            }

            if score_threshold is not None:
                search_params["score_threshold"] = score_threshold

            if filter_conditions:
                search_params["query_filter"] = models.Filter(**filter_conditions)

            results = self.client.search(**search_params)

            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                }
                for result in results
            ]
        except Exception as e:
            raise Exception(f"Failed to search in {collection_name}: {str(e)}")

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection."""
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status.value if info.status else "unknown"
            }
        except Exception as e:
            raise Exception(f"Failed to get collection info for {collection_name}: {str(e)}")

    def count_points(self, collection_name: str) -> int:
        """Count points in a collection."""
        try:
            result = self.client.count(collection_name)
            return result.count
        except Exception as e:
            raise Exception(f"Failed to count points in {collection_name}: {str(e)}")

    def delete_points(
        self,
        collection_name: str,
        point_ids: List[str]
    ) -> bool:
        """Delete specific points from a collection."""
        try:
            # Use the points directly - Qdrant will handle the conversion
            self.client.delete(
                collection_name=collection_name,
                points_selector=point_ids  # type: ignore
            )
            return True
        except Exception as e:
            raise Exception(f"Failed to delete points from {collection_name}: {str(e)}")

    def clear_collection(self, collection_name: str) -> bool:
        """Clear all points from a collection."""
        try:
            # Delete collection and recreate it
            if self.collection_exists(collection_name):
                collection_info = self.client.get_collection(collection_name)

                # Handle different vector config structures
                vectors_config = collection_info.config.params.vectors
                if isinstance(vectors_config, dict):
                    # Get the first (and typically only) vector config
                    vector_config = list(vectors_config.values())[0]
                    vector_size = vector_config.size
                    distance = vector_config.distance
                else:
                    # Direct vector config - fallback to defaults if None
                    vector_size = getattr(vectors_config, 'size', 1536)
                    distance = getattr(vectors_config, 'distance', Distance.COSINE)

                self.client.delete_collection(collection_name)
                self.create_collection(collection_name, vector_size, distance)
            return True
        except Exception as e:
            raise Exception(f"Failed to clear collection {collection_name}: {str(e)}")