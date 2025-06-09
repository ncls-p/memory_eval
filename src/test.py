import abc
import os
import socket
from datetime import datetime
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.prebuilt import create_react_agent
from langgraph.store.base import (
    NOT_PROVIDED,
    BaseStore,
    GetOp,
    ListNamespacesOp,
    NotProvided,
    Op,
    PutOp,
    SearchItem,
    SearchOp,
)
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from mem0 import Memory
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models


class ConfigManager:
    def __init__(self):
        self.ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
        self.qdrant_langchain_collection: str = os.getenv("QDRANT_LANGCHAIN_COLLECTION", "langchain_memories")
        self.qdrant_embedding_dims: int = int(os.getenv("QDRANT_EMBEDDING_DIMS", "1024"))
        self.llm_model_name: str = os.getenv("OLLAMA_MODEL_NAME", "qwen2:1.5b")
        self.embedding_model_name = os.getenv("OLLAMA_EMBEDDING_MODEL_NAME", "mxbai-embed-large")
        self.mem0_qdrant_collection: str = os.getenv("MEM0_QDRANT_COLLECTION", "mem0_memories")


class QdrantConnector:
    def __init__(self, host: str, port: int, timeout: int = 3):
        self.host: str = host
        self.port: int = port
        self.timeout: int = timeout
        self.client: Optional[QdrantClient] = None

    def check_connection(self) -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            return result == 0
        except socket.error as e:
            print(f"Qdrant connection check failed: {e}")
            return False

    def get_client(self) -> Optional[QdrantClient]:
        if not self.check_connection():
            print(f"Warning: Qdrant not available at {self.host}:{self.port}.")
            return None
        if self.client is None:
            try:
                self.client = QdrantClient(host=self.host, port=self.port)
            except Exception as e:
                print(f"Error creating Qdrant client: {e}")
                return None
        return self.client

    def ensure_collection(self, collection_name: str, vector_size: int, distance_model: qdrant_models.Distance = qdrant_models.Distance.COSINE) -> bool:
        client = self.get_client()
        if not client:
            return False
        try:
            client.get_collection(collection_name)
            print(f"Using existing Qdrant collection: {collection_name}")
        except Exception:
            try:
                print(f"Creating new Qdrant collection: {collection_name}")
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=vector_size,
                        distance=distance_model
                    )
                )
            except Exception as e:
                print(f"Error creating Qdrant collection {collection_name}: {e}")
                return False
        return True


class LLMDrivenPersistentStore(BaseStore):
    def __init__(self, base_store: InMemoryStore, vector_store: QdrantVectorStore):
        super().__init__()
        self.base_store: InMemoryStore = base_store
        self.vector_store: QdrantVectorStore = vector_store
        self._load_existing_memories()

    def _load_existing_memories(self):
        try:
            search_results = self.vector_store.similarity_search("", k=1000)
            if search_results:
                print(f"Loaded {len(search_results)} existing memories from Qdrant into LangMem's InMemoryStore.")
                for i, doc in enumerate(search_results):
                    self.base_store.put(
                        ("memories",),
                        f"restored_qdrant_{i}",
                        {"content": doc.page_content, "type": "loaded_from_qdrant"}
                    )
        except Exception as e:
            print(f"Note: Could not load existing memories into LangMem's InMemoryStore: {e}")

    def _persist_to_qdrant(self, namespace: Tuple[str, ...], key: str, value: Any):
        try:
            if isinstance(value, dict):
                content = value.get('content', str(value))
                memory_type = value.get('type', 'general')
                importance = value.get('importance', 'normal')
            else:
                content = str(value)
                memory_type = 'general'
                importance = 'normal'

            metadata = {
                "namespace": str(namespace),
                "key": str(key),
                "content_full": content,
                "memory_type": memory_type,
                "importance": importance,
                "source": "langchain_langmem",
                "persisted_at": datetime.now().isoformat()
            }
            self.vector_store.add_texts([content], metadatas=[metadata])
            print(f"✓ LangMem: Memory persisted to Qdrant: {content[:50]}...")
        except Exception as e:
            print(f"Warning: LangMem: Failed to persist to Qdrant: {e}")

    def put(self, namespace: Tuple[str, ...], key: str, value: Any, index: Optional[Union[List[str], bool]] = None, *, ttl: Union[float, NotProvided, None] = NOT_PROVIDED) -> None:
        actual_index: Optional[Union[List[str], Literal[False]]] = None
        if index is False:
            actual_index = False
        elif isinstance(index, list):
            actual_index = index

        self.base_store.put(namespace, key, value, index=actual_index, ttl=ttl)
        if namespace == ("memories",):
            self._persist_to_qdrant(namespace, key, value)

    def get(self, namespace: Tuple[str, ...], key: str, *, refresh_ttl: Optional[bool] = None) -> Optional[Any]:
        return self.base_store.get(namespace, key, refresh_ttl=refresh_ttl)

    def search(self, namespace_prefix: Tuple[str, ...], *, query: Optional[str] = None, filter: Optional[Dict[str, Any]] = None, limit: int = 10, offset: int = 0, refresh_ttl: Optional[bool] = None) -> List[SearchItem]:
        base_store_results: List[SearchItem] = []
        try:
            base_store_results = self.base_store.search(namespace_prefix, query=query, filter=filter, limit=limit, offset=offset, refresh_ttl=refresh_ttl)
            if base_store_results:
                print(f"LangMem: Found {len(base_store_results)} results from in-memory store component.")
        except Exception as e:
            print(f"LangMem: In-memory store search failed: {e}")

        if query:
            try:
                qdrant_results_with_scores = self.vector_store.similarity_search_with_score(query, k=limit)
                qdrant_search_items: List[SearchItem] = []
                for doc, score in qdrant_results_with_scores:
                    item_key = f"qdrant_{hash(doc.page_content)}_{score}"
                    created_at_str = doc.metadata.get('created_at', doc.metadata.get('persisted_at', datetime.now().isoformat()))
                    updated_at_str = doc.metadata.get('updated_at', doc.metadata.get('persisted_at', datetime.now().isoformat()))

                    try:
                        created_at = datetime.fromisoformat(created_at_str)
                        updated_at = datetime.fromisoformat(updated_at_str)
                    except ValueError:
                        created_at = datetime.now()
                        updated_at = datetime.now()


                    qdrant_search_items.append(SearchItem(
                        namespace=namespace_prefix,
                        key=item_key,
                        value={"content": doc.page_content, "score": score, "source": "qdrant", "metadata": doc.metadata},
                        created_at=created_at,
                        updated_at=updated_at
                    ))

                if qdrant_search_items:
                    print(f"LangMem: Found {len(qdrant_search_items)} results from Qdrant store component.")
                    return qdrant_search_items

            except Exception as e:
                print(f"LangMem: Qdrant search failed: {e}")

        return base_store_results

    def list(self, namespace_prefix: Tuple[str, ...], *, filter: Optional[Dict[str, Any]] = None, limit: Optional[int] = None, offset: int = 0, refresh_ttl: Optional[bool] = None) -> Iterator[SearchItem]:
        effective_limit = limit if limit is not None else -1
        return iter(self.base_store.search(namespace_prefix, query=None, filter=filter, limit=effective_limit, offset=offset, refresh_ttl=refresh_ttl))

    def batch(self, ops: Iterable[Op]) -> List[Any]:
        results = []
        for op_item in ops:
            if isinstance(op_item, PutOp):
                self.put(op_item.namespace, op_item.key, op_item.value, index=op_item.index, ttl=op_item.ttl)
                results.append(None)
            elif isinstance(op_item, GetOp):
                results.append(self.get(op_item.namespace, op_item.key, refresh_ttl=op_item.refresh_ttl))
            elif isinstance(op_item, SearchOp):
                search_limit = op_item.limit if op_item.limit is not None else -1
                results.append(
                    self.search(op_item.namespace_prefix, query=op_item.query, filter=op_item.filter, limit=search_limit, offset=op_item.offset or 0, refresh_ttl=op_item.refresh_ttl)
                )
            elif isinstance(op_item, ListNamespacesOp):
                default_namespace_prefix_for_list_op = ()
                list_limit = -1
                list_offset = 0

                results.append(
                    list(self.list(default_namespace_prefix_for_list_op, filter=None, limit=list_limit, offset=list_offset, refresh_ttl=None))
                )
            else:
                op_type_str = str(getattr(op_item, "type", "unknown_op"))
                try:
                    method = getattr(self.base_store, op_type_str)
                    current_op_namespace = getattr(op_item, 'namespace', getattr(op_item, 'namespace_prefix', None))
                    current_op_key = getattr(op_item, 'key', None)

                    if current_op_namespace is not None and current_op_key is not None:
                         results.append(method(current_op_namespace, current_op_key))
                    elif current_op_namespace is not None:
                         results.append(method(current_op_namespace))
                    else:
                        results.append(method())
                except (AttributeError, TypeError) as e:
                    print(f"Unsupported or malformed batch operation type '{op_type_str}': {e}. Op: {op_item}")
                    results.append(NotImplementedError(f"Operation type '{op_type_str}' not supported in batch."))
        return results

    async def abatch(self, ops: Iterable[Op]) -> List[Any]:
        return self.batch(ops)


class BaseMemorySystem(abc.ABC):
    def __init__(self, config: ConfigManager, system_name: str):
        self.config: ConfigManager = config
        self.system_name: str = system_name
        self.llm: ChatOllama = self._init_llm()
        self.embeddings: OllamaEmbeddings = self._init_embeddings()

    def _init_llm(self, temperature: float = 0.7) -> ChatOllama:
        return ChatOllama(
            model=self.config.llm_model_name,
            base_url=self.config.ollama_base_url,
            temperature=temperature
        )

    def _init_embeddings(self) -> OllamaEmbeddings:
        return OllamaEmbeddings(
            model=self.config.embedding_model_name,
            base_url=self.config.ollama_base_url
        )

    @abc.abstractmethod
    def add(self, messages: List[Dict[str, str]], user_id: Optional[str] = None) -> Any:
        pass

    @abc.abstractmethod
    def search(self, query: str, user_id: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def get_llm_response(self, query: str, memories: List[Dict[str, Any]]) -> str:
        pass

    def format_search_results(self, search_results: List[Dict[str, Any]]) -> None:
        print(f"  Retrieved memories for {self.system_name}:")
        if not search_results:
            print("    No memories found.")
            return
        for i, result in enumerate(search_results, 1):
            content = result.get('content', str(result))
            score = result.get('score', 0.0)
            print(f"    {i}. Memory: {content}")
            if score is not None:
                 print(f"       Confidence: {score:.3f} ({score*100:.1f}%)")
            print()


class LangChainLangMemSystem(BaseMemorySystem):
    def __init__(self, config: ConfigManager, qdrant_connector: QdrantConnector):
        super().__init__(config, "LangChain + LangMem")
        self.qdrant_connector: QdrantConnector = qdrant_connector
        self.store: BaseStore
        self.is_persistent: bool
        self.vector_store: Optional[QdrantVectorStore]
        self.store, self.is_persistent, self.vector_store = self._create_persistent_memory_store()
        self.agent = self._create_agent()

    def _create_persistent_memory_store(self) -> Tuple[BaseStore, bool, Optional[QdrantVectorStore]]:
        default_in_memory_store = InMemoryStore(index={"dims": self.config.qdrant_embedding_dims, "embed": self.embeddings})

        if not self.qdrant_connector.check_connection():
            print(f"Warning ({self.system_name}): Qdrant not available. Falling back to non-persistent InMemoryStore for LangMem.")
            return default_in_memory_store, False, None

        qdrant_client = self.qdrant_connector.get_client()
        if not qdrant_client:
             print(f"Error ({self.system_name}): Failed to get Qdrant client. Falling back to non-persistent InMemoryStore for LangMem.")
             return default_in_memory_store, False, None

        collection_name = self.config.qdrant_langchain_collection
        if not self.qdrant_connector.ensure_collection(collection_name, self.config.qdrant_embedding_dims):
            print(f"Error ({self.system_name}): Failed to ensure Qdrant collection '{collection_name}'. Falling back to non-persistent InMemoryStore for LangMem.")
            return default_in_memory_store, False, None

        try:
            vector_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=collection_name,
                embedding=self.embeddings,
            )
            base_inmemory_store = InMemoryStore(
                index={"dims": self.config.qdrant_embedding_dims, "embed": self.embeddings}
            )
            enhanced_store = LLMDrivenPersistentStore(base_inmemory_store, vector_store)
            print(f"✓ ({self.system_name}): Using persistent Qdrant storage (collection: {collection_name})")
            return enhanced_store, True, vector_store
        except Exception as e:
            print(f"Error ({self.system_name}): Creating persistent store failed: {e}. Falling back to non-persistent InMemoryStore.")
            return default_in_memory_store, False, None

    def _create_agent(self):
        return create_react_agent(
            self.llm,
            tools=[
                create_manage_memory_tool(namespace=("memories",)),
                create_search_memory_tool(namespace=("memories",)),
            ],
            store=self.store,
        )

    def add(self, messages: List[Dict[str, str]], user_id: Optional[str] = None) -> Any:
        print(f"({self.system_name}): Adding messages via agent invocation...")
        formatted_messages = {"messages": messages}
        try:
            return self.agent.invoke(formatted_messages)
        except Exception as e:
            print(f"Error ({self.system_name}): Failed to add messages via agent: {e}")
            return None

    def search(self, query: str, user_id: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        print(f"({self.system_name}): Searching memories for query: '{query}'")
        search_results_items: List[SearchItem] = []
        try:
            search_results_items = self.store.search(
                ("memories",),
                query=query,
                limit=limit
            )
        except Exception as e:
            print(f"Error ({self.system_name}): Store search failed: {e}")

        results_for_formatting = []
        if search_results_items:
            for item in search_results_items:
                content = item.value.get('content', str(item.value)) if isinstance(item.value, dict) else str(item.value)
                score = item.value.get('score', 0.0) if isinstance(item.value, dict) else 0.0
                results_for_formatting.append({"content": content, "score": score})
        return results_for_formatting

    def get_llm_response(self, query: str, memories: List[Dict[str, Any]]) -> str:
        print(f"({self.system_name}): Getting LLM response for query: '{query}'")
        try:
            response = self.agent.invoke({"messages": [{"role": "user", "content": query}]})
            if response and "messages" in response and response["messages"]:
                last_message = response["messages"][-1]
                if hasattr(last_message, 'content'):
                    return last_message.content
                elif isinstance(last_message, dict) and 'content' in last_message:
                    return last_message['content']
            return "Error: Could not parse LLM response from agent. No suitable content found."
        except Exception as e:
            print(f"Error ({self.system_name}): Agent invocation for LLM response failed: {e}")
            return f"Error generating response: {e}"


class Mem0System(BaseMemorySystem):
    def __init__(self, config: ConfigManager, qdrant_connector: QdrantConnector, collection_name: Optional[str] = None):
        super().__init__(config, "Mem0")
        self.qdrant_connector: QdrantConnector = qdrant_connector
        self.collection_name = collection_name or self.config.mem0_qdrant_collection
        self.mem0_instance: Optional[Memory] = self._init_mem0()

    def _get_mem0_config(self, use_qdrant: bool) -> Dict[str, Any]:
        base_config = {
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": self.config.llm_model_name,
                    "ollama_base_url": self.config.ollama_base_url,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": self.config.embedding_model_name,
                    "ollama_base_url": self.config.ollama_base_url,
                },
            },
            "vector_store": {}
        }

        if use_qdrant:
            collection_ensured = self.qdrant_connector.ensure_collection(
                self.collection_name,
                self.config.qdrant_embedding_dims
            )
            if collection_ensured:
                base_config["vector_store"] = {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": self.collection_name,
                        "host": self.config.qdrant_host,
                        "port": self.config.qdrant_port,
                    },
                }
                print(f"✓ ({self.system_name}): Configured to use Qdrant (collection: {self.collection_name})")
            else:
                print(f"⚠ ({self.system_name}): Failed to ensure Qdrant collection for Mem0. Falling back.")
                use_qdrant = False

        if not use_qdrant:
            base_config["vector_store"] = {
                "provider": "chroma",
                "config": {
                    "collection_name": "mem0_chroma_default",
                    "path": "./chroma_db_mem0",
                },
            }
            print(f"⚠ ({self.system_name}): Using fallback vector store (Chroma) for Mem0.")
        return base_config

    def _init_mem0(self) -> Optional[Memory]:
        qdrant_available_and_checked = self.qdrant_connector.check_connection()
        mem0_config = self._get_mem0_config(use_qdrant=qdrant_available_and_checked)
        try:
            return Memory.from_config(mem0_config)
        except Exception as e:
            print(f"Error ({self.system_name}): Failed to initialize Mem0 from config: {e}")
            print(f"Mem0 config used: {mem0_config}")
            return None

    def add(self, messages: List[Dict[str, str]], user_id: Optional[str] = "default_user") -> Any:
        if not self.mem0_instance:
            print(f"Error ({self.system_name}): Mem0 instance not initialized. Cannot add messages.")
            return None
        print(f"({self.system_name}): Adding messages for user_id: {user_id}...")
        try:
            return self.mem0_instance.add(messages, user_id=user_id)
        except Exception as e:
            print(f"Error ({self.system_name}): Failed to add messages to Mem0: {e}")
            return None

    def search(self, query: str, user_id: Optional[str] = "default_user", limit: int = 5) -> List[Dict[str, Any]]:
        if not self.mem0_instance:
            print(f"Error ({self.system_name}): Mem0 instance not initialized. Cannot search.")
            return []

        print(f"({self.system_name}): Searching memories for query: '{query}', user_id: {user_id}")
        formatted_memories = []
        try:
            memories_response = self.mem0_instance.search(query, user_id=user_id, limit=limit)

            raw_memories_list = []
            if isinstance(memories_response, list):
                raw_memories_list = memories_response
            elif isinstance(memories_response, dict) and 'results' in memories_response and isinstance(memories_response['results'], list):
                raw_memories_list = memories_response['results']

            for mem_data in raw_memories_list:
                if isinstance(mem_data, dict):
                    content = mem_data.get('memory', mem_data.get('text', str(mem_data)))
                    score = mem_data.get('score', mem_data.get('similarity', mem_data.get('confidence')))
                else:
                    content = str(mem_data)
                    score = None
                formatted_memories.append({"content": content, "score": score if score is not None else 0.0})

        except Exception as e:
            print(f"Error ({self.system_name}): Mem0 search failed: {e}")

        return formatted_memories[:limit]

    def get_llm_response(self, query: str, memories: List[Dict[str, Any]]) -> str:
        print(f"({self.system_name}): Getting LLM response for query: '{query}'")
        if not memories:
            context_str = "No relevant memories found for the user."
        else:
            context_str = "\n".join([f"- {memory.get('content', str(memory))}" for memory in memories])

        prompt = f"""Based on the following information retrieved from the user's memory:
{context_str}

User's current question: {query}

Please provide a helpful and concise response based *only* on the provided memories and the question. If the memories do not contain relevant information, state that.
"""
        try:
            response = self.llm.invoke(prompt)
            llm_output = response.content
            if isinstance(llm_output, str):
                return llm_output
            elif isinstance(llm_output, list):
                return " ".join(str(p) for p in llm_output if p is not None)
            else:
                return str(llm_output)
        except Exception as e:
            print(f"Error ({self.system_name}): LLM invocation failed: {e}")
            return f"Error generating response: {e}"


class MemoryTester:
    def __init__(self, config: ConfigManager, qdrant_connector: QdrantConnector):
        self.config: ConfigManager = config
        self.qdrant_connector: QdrantConnector = qdrant_connector
        self.systems: List[BaseMemorySystem] = []

    def add_system(self, system: BaseMemorySystem):
        self.systems.append(system)

    def run_test_on_system(self, system: BaseMemorySystem, first_message_content: str, question_content: str, user_id: str = "test_user_main"):
        print(f"\n{'=' * 20} Testing: {system.system_name} {'=' * 20}")

        initial_messages = [{"role": "user", "content": first_message_content}]

        print(f"\n1. Adding initial message to {system.system_name}: '{first_message_content}' (User ID: {user_id})")
        add_result = system.add(initial_messages, user_id=user_id)
        print(f"   Add operation result (type: {type(add_result)}): {str(add_result)[:200]}...")


        print(f"\n2. Searching memories in {system.system_name} with question: '{question_content}' (User ID: {user_id})")
        retrieved_memories = system.search(question_content, user_id=user_id, limit=5)
        system.format_search_results(retrieved_memories)

        print(f"\n3. Generating LLM response from {system.system_name} for question: '{question_content}'")
        llm_response = system.get_llm_response(question_content, retrieved_memories)
        print(f"\n{system.system_name} Response:\n{llm_response}")
        print(f"{'-' * (40 + len(system.system_name))}")

    def run_comparison(self, first_message_content: str, question_content: str):
        print("=" * 60)
        print("MEMORY SYSTEMS COMPARISON: LangChain+LangMem vs Mem0 (with Ollama)")
        print("=" * 60)

        langchain_system = LangChainLangMemSystem(self.config, self.qdrant_connector)
        mem0_system = Mem0System(self.config, self.qdrant_connector)

        self.add_system(langchain_system)
        self.add_system(mem0_system)

        if not self.systems:
            print("No memory systems configured for testing.")
            return

        for system in self.systems:
            self.run_test_on_system(system, first_message_content, question_content, user_id=f"user_{system.system_name.replace(' + ', '_').lower()}")

        print("\n" + "=" * 60)
        print("Comparison finished.")
        print("=" * 60)


def main():
    load_dotenv()

    first_message = "I enjoy watching historical documentaries, especially those about ancient Rome. I usually watch them in the evening around 8 PM."
    question = "What kind of documentaries do I like and when do I watch them?"

    print("Initializing configuration and connectors...")
    config = ConfigManager()
    qdrant_connector = QdrantConnector(host=config.qdrant_host, port=config.qdrant_port)

    print("\nChecking Qdrant availability...")
    if qdrant_connector.check_connection():
        print("Qdrant service is accessible.")
    else:
        print("Warning: Qdrant service is NOT accessible. Persistence and vector search capabilities will be limited or unavailable for systems relying on it.")

    tester = MemoryTester(config, qdrant_connector)
    tester.run_comparison(first_message_content=first_message, question_content=question)


if __name__ == "__main__":
    main()
