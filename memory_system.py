# # from sentence_transformers import SentenceTransformer

# # External Imports
# import os
# import json
# import uuid
# import redis
# import faiss
# import numpy as np
# from datetime import datetime
# from pydantic import BaseModel


# # Typings
# class GenerateTopics(BaseModel):
#     topics_list: list[str]


# class MemoraBrain:
#     def __init__(self, 
#                 langchain_llm,
#                 langchain_embedding_model,
#                 host='localhost', 
#                 port=6379, 
#                 db=0, 
#                 vecdb_base_path="vecdbs"):

#         # initialize redis client to connect to in-memory DB
#         self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)

#         # creating a folder for storing vectorDBs of threads
#         self.vecdb_base_path = vecdb_base_path
        
#         # initializing langchain llm and emebdding models, user's get to choose the model and then they integrate it
#         self._llm = langchain_llm
#         self._embedding_model = langchain_embedding_model
#         self._embedding_model_name = str(langchain_embedding_model.model) # storing model name
#         # print("PRINTING IN INITALIZATION: {model_name}".format(model_name = str(langchain_embedding_model.model)))

#         # Create vecdb directory if it doesn't exist
#         if not os.path.exists(vecdb_base_path):
#             os.makedirs(vecdb_base_path, exist_ok=True)

    
#     # ========== Embedding Generation ==========    
#     def generateEmbeddingsViaLangchain(self, text):
#         # generating and typecasting embeddings to numpy float32
#         embedding = self._embedding_model.embed_query(text)
#         embedding = np.array(embedding, dtype=np.float32)
#         return embedding
    
#     # ========== Topic Generation ==========    
#     def generateOrMatchTopicsViaLangchain(self, question, answer, existingTopics):       
#         alreadyExistingTopics = existingTopics
#         prompt = F"""
# Classify the following conversation into very generic topic names (like sports, food, math, movies etc). They can either be new, or be an overlap of already existing topics.
# already existing topics: {alreadyExistingTopics}
# Q: {question}
# A: {answer}
# Only return a list separated by commas.
# """
#         response = self._llm.with_structured_output(GenerateTopics).invoke(prompt)
#         return response.topics_list
        
    
#     # ========== Conversation Window Management ==========
#     def createConversationWindow(self, query_text: str, answer_text: str, thread_id: str, referral_ids=None, user_name=None, generate_topics=True):
#         # Create a new context window and store it in Redis
        
#         assert thread_id, "ThreadId is missing"
#         assert isinstance(query_text, str) and query_text.strip(), "Query text is missing"
#         assert isinstance(answer_text, str) and answer_text.strip(), "Answer text is missing"
#         assert self._embedding_model_name == self.getThreadsEmbeddingModelName(threadId=thread_id), f"Different Embedding model's used, revert back to the initial one [i.e this model --> {self.getThreadsEmbeddingModelName(threadId=thread_id)}]"

#         # pre-processing text
#         query_text = query_text.strip()
#         answer_text = answer_text.strip()

#         # Generate a UUID for the window
#         window_id = str(uuid.uuid4())

#         # LOGs
#         print("WINDOW ID GENERATED")

#         # Calculate timestamp
#         timestamp = datetime.now().timestamp()

#         # Generate topics if requested
#         topics = []
#         if generate_topics:
#             alreadyExistingTopicsinThread = self.getThreadTopics(thread_id)
#             topics = self.generateOrMatchTopicsViaLangchain(question = query_text, 
#                                                             answer = answer_text, 
#                                                             existingTopics= (alreadyExistingTopicsinThread if len(alreadyExistingTopicsinThread) > 0 else "No topics stored yet."))
#             print("TOPICS:", topics)
            
        
#         # Store the context window as a hash
#         self.redis_client.hset(
#             f"window:{window_id}",
#             mapping={
#                 "window_id": window_id,
#                 "query_text": query_text,
#                 "answer_text": answer_text,
#                 "referral_ids": ','.join(referral_ids) if referral_ids else '',
#                 "timestamp": timestamp,
#                 "user_name": user_name or '',
#                 "topics": ','.join(topics)
#             }
#         )

#         print("WINDOW STORED IN REDIS")

#         # Generate embedding for the Q&A pair
#         embedding = self.generateEmbeddingsViaLangchain(query_text + " " + answer_text)


#         print("EMBEDDINGS CREATED")

#         # Thread or conversation it belongs to
#         THREAD_ID = thread_id

#         # Add to thread
#         self.addWindow2Thread(thread_id, window_id, topics)

#         # have to insert this to vector DB
#         _ = self.create_vecdb_if_needed(thread_id, dim=len(embedding))

#         # meta data and vector upsertion
#         metadata = {
#             "window_id": window_id,
#             "topics": topics,
#             "timestamp": str(timestamp),
#             "username": user_name,
#             "question": query_text,
#             "answer": answer_text
#         }
        
#         vector_id = self.insert_into_vecdb(thread_id, embedding, metadata)

#         print("UPSERTED VECTORS")

#         return {
#             "window_id": window_id,
#             "thread_id": thread_id,
#             "vector_id": vector_id,
#             "topics": topics
#         }


#     # ========== Thread Management ==========
    
#     def createThread(self): # a new thread is created and its id is returned.
#         thread_id = f"Thread_{uuid.uuid4()}"
#         self.redis_client.hset(thread_id, "all_windows", json.dumps([]))
#         self.redis_client.hset(thread_id, "embedding_model_used", self._embedding_model_name)
#         return thread_id
    
#     def getThreadsEmbeddingModelName(self, threadId): # returns the embedding model's name that was used when the thread was created.
#         return self.redis_client.hget(threadId, "embedding_model_used")

#     def getThreadTopics(self, threadId): # returns a list of all topics that the thread has.
#         all_keys_in_thread = [key for key in self.redis_client.hkeys(threadId)] # parsing keys into an array
#         all_topics = [k for k in all_keys_in_thread if k not in ("all_windows", "embedding_model_used")]
#         return all_topics
    
#     def addWindow2Thread(self, thread_id, window_id, topics): # used to add conversation windows to threads.
#         # redis pipe for easier transaction
#         # pipe = self.redis_client.pipeline()

#         # Add to topic-specific lists
#         for topic in topics:
#             topic_reference = self.redis_client.hget(thread_id, topic)
#             if topic_reference: # if topic exists
#                 existing = json.loads(topic_reference)
#                 existing.append(window_id)
#                 self.redis_client.hset(thread_id, topic, json.dumps(existing))
#             else: # if topic doesnt exist create a new one
#                 self.redis_client.hset(thread_id, topic, json.dumps([window_id]))

#         # get all windows from the directory
#         all_windows = json.loads(self.redis_client.hget(thread_id, "all_windows") or "[]")
#         all_windows.append(window_id) # add current window id to all windows
#         self.redis_client.hset(thread_id, "all_windows", json.dumps(all_windows)) # set the updated all windows back again
        
#     def getThreadDetails(self, thread_id): # all details of a threadid is returned
#         thread_data = self.redis_client.hgetall(thread_id)
        
#         if thread_data:
#             # Convert comma-separated window IDs back to lists
#             return thread_data
#         return None

#     # ========== Vector DB Functionality ==========

#     def create_vecdb_if_needed(self, thread_id, dim):
#         """Create a cosine similarity-based vector database if it doesn't exist"""
#         vecdb_path = os.path.join(self.vecdb_base_path, thread_id)
        
#         # Create directory if it doesn't exist
#         if not os.path.exists(vecdb_path):
#             os.makedirs(vecdb_path, exist_ok=True)
            
#         # Create index file if it doesn't exist
#         index_path = os.path.join(vecdb_path, "index.faiss")
#         if not os.path.exists(index_path):
#             # Important: use IndexFlatIP (inner product) + normalize vectors
#             index = faiss.IndexFlatIP(dim)  # <---- use IP (Inner Product)
#             faiss.write_index(index, index_path)
        
#         # Create metadata file if it doesn't exist
#         metadata_path = os.path.join(vecdb_path, "metadata.json")
#         if not os.path.exists(metadata_path):
#             with open(metadata_path, 'w') as f:
#                 json.dump({}, f)

    
#     def insert_into_vecdb(self, thread_id, embedding, metadata):
#         vecdb_path = os.path.join(self.vecdb_base_path, thread_id)
#         index_path = os.path.join(vecdb_path, "index.faiss")
#         metadata_path = os.path.join(vecdb_path, "metadata.json")

#         if not os.path.exists(index_path) or not os.path.exists(metadata_path):
#             raise FileNotFoundError("Index or metadata file missing. Did you run create_vecdb_if_needed?")

#         index = faiss.read_index(index_path)

#         # Ensure embedding is properly shaped for FAISS
#         if not isinstance(embedding, np.ndarray):
#             embedding = np.array(embedding, dtype=np.float32)
        
#         if embedding.ndim == 1:
#             embedding = np.array([embedding], dtype=np.float32)
        
#         # Verify dimensions match
#         if embedding.shape[1] != index.d:
#             raise ValueError(f"Embedding dimensionality {embedding.shape[1]} does not match index dimensionality {index.d}.")

#         index.add(embedding)

#         # Load and update metadata
#         try:
#             with open(metadata_path, 'r+') as f:
#                 try:
#                     data = json.load(f)
#                 except json.JSONDecodeError:
#                     data = {}

#                 vector_id = str(index.ntotal - 1)  # ID of the embedding we just added
#                 data[vector_id] = metadata
                
#                 f.seek(0)
#                 json.dump(data, f, indent=2)
#                 f.truncate()
#         except FileNotFoundError:
#             raise FileNotFoundError("Metadata file not found.")

#         # Save updated index
#         faiss.write_index(index, index_path)
        
#         return vector_id
    
#     # def search_by_text(self, search_text, limit=10):
#     #     """Search context windows for text in query or answer"""
#     #     matching_windows = []
        
#     #     # Get all keys that are context windows
#     #     window_keys = self.redis_client.keys("window:*")
        
#     #     for key in window_keys:
#     #         window_data = self.redis_client.hgetall(key)
            
#     #         # Check if search text is in the query or answer text
#     #         query_text = window_data.get("query_text", "").lower()
#     #         answer_text = window_data.get("answer_text", "").lower()
            
#     #         if search_text.lower() in query_text or search_text.lower() in answer_text:
#     #             # Process referral_ids
#     #             if 'referral_ids' in window_data and window_data['referral_ids']:
#     #                 window_data['referral_ids'] = window_data['referral_ids'].split(',')
#     #             else:
#     #                 window_data['referral_ids'] = []
                
#     #             # Process topics
#     #             if 'topics' in window_data and window_data['topics']:
#     #                 window_data['topics'] = window_data['topics'].split(',')
#     #             else:
#     #                 window_data['topics'] = []
                    
#     #             # Convert timestamp to float
#     #             if 'timestamp' in window_data:
#     #                 window_data['timestamp'] = float(window_data['timestamp'])
                    
#     #             matching_windows.append(window_data)
                
#     #             # Limit results
#     #             if len(matching_windows) >= limit:
#     #                 break
                    
#     #     return matching_windows
    
#     # def search_by_user(self, user_name, limit=10):
#     #     """Search for context windows by user name"""
#     #     matching_windows = []
        
#     #     # Get all keys that are context windows
#     #     window_keys = self.redis_client.keys("window:*")
        
#     #     for key in window_keys:
#     #         window_data = self.redis_client.hgetall(key)
            
#     #         # Check if user name matches
#     #         if window_data.get("user_name") == user_name:
#     #             # Process referral_ids
#     #             if 'referral_ids' in window_data and window_data['referral_ids']:
#     #                 window_data['referral_ids'] = window_data['referral_ids'].split(',')
#     #             else:
#     #                 window_data['referral_ids'] = []
                
#     #             # Process topics
#     #             if 'topics' in window_data and window_data['topics']:
#     #                 window_data['topics'] = window_data['topics'].split(',')
#     #             else:
#     #                 window_data['topics'] = []
                    
#     #             # Convert timestamp to float
#     #             if 'timestamp' in window_data:
#     #                 window_data['timestamp'] = float(window_data['timestamp'])
                    
#     #             matching_windows.append(window_data)
                
#     #             # Limit results
#     #             if len(matching_windows) >= limit:
#     #                 break
                    
#     #     return matching_windows
    
#     # def semantic_search(self, query_text, thread_id=None, top_k=5):
#     #     """Perform semantic search using vector database"""
#     #     # Get the embedding for the query
#     #     query_embedding = self.get_embedding(query_text)
        
#     #     # If thread_id is provided, search only in that thread
#     #     if thread_id:
#     #         return self.search_in_vecdb(thread_id, query_embedding, top_k)
        
#     #     # Otherwise, search in all vector databases
#     #     results = []
#     #     thread_dirs = [d for d in os.listdir(self.vecdb_base_path) 
#     #                     if os.path.isdir(os.path.join(self.vecdb_base_path, d))]
        
#     #     for thread_dir in thread_dirs:
#     #         thread_results = self.search_in_vecdb(thread_dir, query_embedding, top_k)
#     #         if thread_results:
#     #             # Include thread_id in results
#     #             for result in thread_results:
#     #                 result["thread_id"] = thread_dir
#     #             results.extend(thread_results)
        
#     #     # Sort by distance
#     #     results.sort(key=lambda x: x["distance"])
        
#     #     # Limit to top_k overall results
#     #     return results[:top_k]
    
#     # def search_in_vecdb(self, thread_id, query_embedding, top_k=5):
#         """Search in a specific vector database"""
#         vecdb_path = os.path.join(self.vecdb_base_path, thread_id)
#         index_path = os.path.join(vecdb_path, "index.faiss")
#         metadata_path = os.path.join(vecdb_path, "metadata.json")
        
#         if not os.path.exists(index_path) or not os.path.exists(metadata_path):
#             return []
            
#         # Load the index
#         index = faiss.read_index(index_path)
        
#         # Ensure proper shape
#         if query_embedding.ndim == 1:
#             query_embedding = np.array([query_embedding], dtype=np.float32)
            
#         # Search
#         distances, indices = index.search(query_embedding, top_k)
        
#         # Load metadata
#         with open(metadata_path, 'r') as f:
#             metadata = json.load(f)
            
#         # Format results
#         results = []
#         for i, idx in enumerate(indices[0]):
#             if idx == -1:  # FAISS returns -1 if fewer than top_k results found
#                 continue
                
#             # Get the metadata for this vector
#             vector_id = str(idx)
#             if vector_id in metadata:
#                 result = metadata[vector_id].copy()
#                 result["distance"] = float(distances[0][i])
#                 result["vector_id"] = vector_id
#                 results.append(result)
                
#         return results















# from sentence_transformers import SentenceTransformer

# External Imports
import os
import json
import uuid
import redis
import faiss
import numpy as np
from datetime import datetime
from pydantic import BaseModel


# Typings
class GenerateTopics(BaseModel):
    topics_list: list[str]


class MemoraBrain:
    def __init__(self, 
                langchain_llm,
                langchain_embedding_model,
                host='localhost', 
                port=6379, 
                db=0, 
                vecdb_base_path="vecdbs"):

        # initialize redis client to connect to in-memory DB
        self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)

        # creating a folder for storing vectorDBs of threads
        self.vecdb_base_path = vecdb_base_path
        
        # initializing langchain llm and emebdding models, user's get to choose the model and then they integrate it
        self._llm = langchain_llm
        self._embedding_model = langchain_embedding_model
        self._embedding_model_name = str(langchain_embedding_model.model) # storing model name
        # print("PRINTING IN INITALIZATION: {model_name}".format(model_name = str(langchain_embedding_model.model)))

        # Create vecdb directory if it doesn't exist
        if not os.path.exists(vecdb_base_path):
            os.makedirs(vecdb_base_path, exist_ok=True)

    
    # ========== Embedding Generation ==========    
    def generateEmbeddingsViaLangchain(self, text):
        # generating and typecasting embeddings to numpy float32
        embedding = self._embedding_model.embed_query(text)
        embedding = np.array(embedding, dtype=np.float32)
        return embedding
    
    # ========== Topic Generation ==========    
    def generateOrMatchTopicsViaLangchain(self, question, answer, existingTopics):       
        alreadyExistingTopics = existingTopics
        prompt = F"""
Classify the following conversation into very generic topic names (like sports, food, math, movies etc). They can either be new, or be an overlap of already existing topics.
already existing topics: {alreadyExistingTopics}
Q: {question}
A: {answer}
Only return a list separated by commas.
"""
        response = self._llm.with_structured_output(GenerateTopics).invoke(prompt)
        return response.topics_list
        
    
    # ========== Conversation Window Management ==========
    def createConversationWindow(self, query_text: str, answer_text: str, thread_id: str, referral_ids=None, user_name=None, generate_topics=True):
        # Create a new context window and store it in Redis
        
        assert thread_id, "ThreadId is missing"
        assert isinstance(query_text, str) and query_text.strip(), "Query text is missing"
        assert isinstance(answer_text, str) and answer_text.strip(), "Answer text is missing"
        assert self._embedding_model_name == self.getThreadsEmbeddingModelName(threadId=thread_id), f"Different Embedding model's used, revert back to the initial one [i.e this model --> {self.getThreadsEmbeddingModelName(threadId=thread_id)}]"

        # pre-processing text
        query_text = query_text.strip()
        answer_text = answer_text.strip()

        # Generate a UUID for the window
        window_id = str(uuid.uuid4())

        # LOGs
        print("WINDOW ID GENERATED")

        # Calculate timestamp
        timestamp = datetime.now().timestamp()

        # Generate topics if requested
        topics = []
        if generate_topics:
            alreadyExistingTopicsinThread = self.getThreadTopics(thread_id)
            topics = self.generateOrMatchTopicsViaLangchain(question = query_text, 
                                                            answer = answer_text, 
                                                            existingTopics= (alreadyExistingTopicsinThread if len(alreadyExistingTopicsinThread) > 0 else "No topics stored yet."))
            print("TOPICS:", topics)
            
        
        # Store the context window as a hash
        self.redis_client.hset(
            f"window:{window_id}",
            mapping={
                "window_id": window_id,
                "query_text": query_text,
                "answer_text": answer_text,
                "referral_ids": ','.join(referral_ids) if referral_ids else '',
                "timestamp": timestamp,
                "user_name": user_name or '',
                "topics": ','.join(topics)
            }
        )

        print("WINDOW STORED IN REDIS")

        # Generate embedding for the Q&A pair
        embedding = self.generateEmbeddingsViaLangchain(query_text + " " + answer_text)


        print("EMBEDDINGS CREATED")

        # Thread or conversation it belongs to
        THREAD_ID = thread_id

        # Add to thread
        self.addWindow2Thread(thread_id, window_id, topics)

        # have to insert this to vector DB
        _ = self.create_vecdb_if_needed(thread_id, dim=len(embedding))

        # meta data and vector upsertion
        metadata = {
            "window_id": window_id,
            "topics": topics,
            "timestamp": str(timestamp),
            "username": user_name,
            "question": query_text,
            "answer": answer_text
        }
        
        vector_id = self.insert_into_vecdb(thread_id, embedding, metadata)

        print("UPSERTED VECTORS")

        return {
            "window_id": window_id,
            "thread_id": thread_id,
            "vector_id": vector_id,
            "topics": topics
        }


    # ========== Thread Management ==========
    
    def createThread(self): # a new thread is created and its id is returned.
        thread_id = f"Thread_{uuid.uuid4()}"
        self.redis_client.hset(thread_id, "all_windows", json.dumps([]))
        self.redis_client.hset(thread_id, "embedding_model_used", self._embedding_model_name)
        return thread_id
    
    def getThreadsEmbeddingModelName(self, threadId): # returns the embedding model's name that was used when the thread was created.
        return self.redis_client.hget(threadId, "embedding_model_used")

    def getThreadTopics(self, threadId): # returns a list of all topics that the thread has.
        all_keys_in_thread = [key for key in self.redis_client.hkeys(threadId)] # parsing keys into an array
        all_topics = [k for k in all_keys_in_thread if k not in ("all_windows", "embedding_model_used")]
        return all_topics
    
    def addWindow2Thread(self, thread_id, window_id, topics): # used to add conversation windows to threads.
        # redis pipe for easier transaction
        # pipe = self.redis_client.pipeline()

        # Add to topic-specific lists
        for topic in topics:
            topic_reference = self.redis_client.hget(thread_id, topic)
            if topic_reference: # if topic exists
                existing = json.loads(topic_reference)
                existing.append(window_id)
                self.redis_client.hset(thread_id, topic, json.dumps(existing))
            else: # if topic doesnt exist create a new one
                self.redis_client.hset(thread_id, topic, json.dumps([window_id]))

        # get all windows from the directory
        all_windows = json.loads(self.redis_client.hget(thread_id, "all_windows") or "[]")
        all_windows.append(window_id) # add current window id to all windows
        self.redis_client.hset(thread_id, "all_windows", json.dumps(all_windows)) # set the updated all windows back again
        
    def getThreadDetails(self, thread_id): # all details of a threadid is returned
        thread_data = self.redis_client.hgetall(thread_id)
        
        if thread_data:
            # Convert comma-separated window IDs back to lists
            return thread_data
        return None

    # ========== Vector DB Functionality ==========

    def create_vecdb_if_needed(self, thread_id, dim):
        """Create a cosine similarity-based vector database if it doesn't exist"""
        vecdb_path = os.path.join(self.vecdb_base_path, thread_id)
        
        # Create directory if it doesn't exist
        if not os.path.exists(vecdb_path):
            os.makedirs(vecdb_path, exist_ok=True)
            
        # Create index file if it doesn't exist
        index_path = os.path.join(vecdb_path, "index.faiss")
        if not os.path.exists(index_path):
            # Important: use IndexFlatIP (inner product) + normalize vectors
            index = faiss.IndexFlatIP(dim)  # <---- use IP (Inner Product)
            faiss.write_index(index, index_path)
        
        # Create metadata file if it doesn't exist
        metadata_path = os.path.join(vecdb_path, "metadata.json")
        if not os.path.exists(metadata_path):
            with open(metadata_path, 'w') as f:
                json.dump({}, f)

    
    def insert_into_vecdb(self, thread_id, embedding, metadata):
        vecdb_path = os.path.join(self.vecdb_base_path, thread_id)
        index_path = os.path.join(vecdb_path, "index.faiss")
        metadata_path = os.path.join(vecdb_path, "metadata.json")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("Index or metadata file missing. Did you run create_vecdb_if_needed?")

        index = faiss.read_index(index_path)

        # Ensure embedding is properly shaped for FAISS
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        if embedding.ndim == 1:
            embedding = np.array([embedding], dtype=np.float32)
        
        # Verify dimensions match
        if embedding.shape[1] != index.d:
            raise ValueError(f"Embedding dimensionality {embedding.shape[1]} does not match index dimensionality {index.d}.")

        index.add(embedding)

        # Load and update metadata
        try:
            with open(metadata_path, 'r+') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}

                vector_id = str(index.ntotal - 1)  # ID of the embedding we just added
                data[vector_id] = metadata
                
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
        except FileNotFoundError:
            raise FileNotFoundError("Metadata file not found.")

        # Save updated index
        faiss.write_index(index, index_path)
        
        return vector_id
    
    # def search_by_text(self, search_text, limit=10):
    #     """Search context windows for text in query or answer"""
    #     matching_windows = []
        
    #     # Get all keys that are context windows
    #     window_keys = self.redis_client.keys("window:*")
        
    #     for key in window_keys:
    #         window_data = self.redis_client.hgetall(key)
            
    #         # Check if search text is in the query or answer text
    #         query_text = window_data.get("query_text", "").lower()
    #         answer_text = window_data.get("answer_text", "").lower()
            
    #         if search_text.lower() in query_text or search_text.lower() in answer_text:
    #             # Process referral_ids
    #             if 'referral_ids' in window_data and window_data['referral_ids']:
    #                 window_data['referral_ids'] = window_data['referral_ids'].split(',')
    #             else:
    #                 window_data['referral_ids'] = []
                
    #             # Process topics
    #             if 'topics' in window_data and window_data['topics']:
    #                 window_data['topics'] = window_data['topics'].split(',')
    #             else:
    #                 window_data['topics'] = []
                    
    #             # Convert timestamp to float
    #             if 'timestamp' in window_data:
    #                 window_data['timestamp'] = float(window_data['timestamp'])
                    
    #             matching_windows.append(window_data)
                
    #             # Limit results
    #             if len(matching_windows) >= limit:
    #                 break
                    
    #     return matching_windows
    
    # def search_by_user(self, user_name, limit=10):
    #     """Search for context windows by user name"""
    #     matching_windows = []
        
    #     # Get all keys that are context windows
    #     window_keys = self.redis_client.keys("window:*")
        
    #     for key in window_keys:
    #         window_data = self.redis_client.hgetall(key)
            
    #         # Check if user name matches
    #         if window_data.get("user_name") == user_name:
    #             # Process referral_ids
    #             if 'referral_ids' in window_data and window_data['referral_ids']:
    #                 window_data['referral_ids'] = window_data['referral_ids'].split(',')
    #             else:
    #                 window_data['referral_ids'] = []
                
    #             # Process topics
    #             if 'topics' in window_data and window_data['topics']:
    #                 window_data['topics'] = window_data['topics'].split(',')
    #             else:
    #                 window_data['topics'] = []
                    
    #             # Convert timestamp to float
    #             if 'timestamp' in window_data:
    #                 window_data['timestamp'] = float(window_data['timestamp'])
                    
    #             matching_windows.append(window_data)
                
    #             # Limit results
    #             if len(matching_windows) >= limit:
    #                 break
                    
    #     return matching_windows
    
    # def semantic_search(self, query_text, thread_id=None, top_k=5):
    #     """Perform semantic search using vector database"""
    #     # Get the embedding for the query
    #     query_embedding = self.get_embedding(query_text)
        
    #     # If thread_id is provided, search only in that thread
    #     if thread_id:
    #         return self.search_in_vecdb(thread_id, query_embedding, top_k)
        
    #     # Otherwise, search in all vector databases
    #     results = []
    #     thread_dirs = [d for d in os.listdir(self.vecdb_base_path) 
    #                     if os.path.isdir(os.path.join(self.vecdb_base_path, d))]
        
    #     for thread_dir in thread_dirs:
    #         thread_results = self.search_in_vecdb(thread_dir, query_embedding, top_k)
    #         if thread_results:
    #             # Include thread_id in results
    #             for result in thread_results:
    #                 result["thread_id"] = thread_dir
    #             results.extend(thread_results)
        
    #     # Sort by distance
    #     results.sort(key=lambda x: x["distance"])
        
    #     # Limit to top_k overall results
    #     return results[:top_k]
    
    # def search_in_vecdb(self, thread_id, query_embedding, top_k=5):
        """Search in a specific vector database"""
    ''' vecdb_path = os.path.join(self.vecdb_base_path, thread_id)
        index_path = os.path.join(vecdb_path, "index.faiss")
        metadata_path = os.path.join(vecdb_path, "metadata.json")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            return []
            
        # Load the index
        index = faiss.read_index(index_path)
        
        # Ensure proper shape
        if query_embedding.ndim == 1:
            query_embedding = np.array([query_embedding], dtype=np.float32)
            
        # Search
        distances, indices = index.search(query_embedding, top_k)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS returns -1 if fewer than top_k results found
                continue
                
            # Get the metadata for this vector
            vector_id = str(idx)
            if vector_id in metadata:
                result = metadata[vector_id].copy()
                result["distance"] = float(distances[0][i])
                result["vector_id"] = vector_id
                results.append(result)
                
        return results '''
    
    
    def search_by_question(self, question_text, thread_id, top_k=5):
        """
        Search for relevant conversation windows based on a new question using FAISS and metadata.
        """
        import os
        import json
        import numpy as np
        import faiss

        # --- Validation ---
        if not question_text or not isinstance(question_text, str):
            raise ValueError("A valid question text is required.")
        if not thread_id:
            raise ValueError("Thread ID is required.")

        # --- Check embedding model consistency ---
        expected_model = self.getThreadsEmbeddingModelName(threadId=thread_id)
        if self._embedding_model_name != expected_model:
            raise ValueError(f"Inconsistent embedding model. Expected: {expected_model}")

        # --- Step 1: Generate Topics ---
        existing_topics = self.getThreadTopics(thread_id)
        question_topics = self.generateOrMatchTopicsViaLangchain(
            question=question_text,
            answer="",
            existingTopics=existing_topics or "No topics stored yet."
        )
        print(f"Generated topics: {question_topics}")

        # --- Step 2: Get Matching Window IDs ---
        thread_details = self.getThreadDetails(thread_id)
        if not thread_details:
            print(f"No thread details found for thread ID: {thread_id}")
            return []

        matching_window_ids = set()
        for topic in question_topics:
            topic_windows = thread_details.get(topic)
            if topic_windows:
                matching_window_ids.update(json.loads(topic_windows))

        # If no matches found, fallback to all windows
        if not matching_window_ids:
            matching_window_ids.update(json.loads(thread_details.get("all_windows", "[]")))

        if not matching_window_ids:
            print("No conversation windows found in thread.")
            return []

        # --- Step 3: Vectorize Question ---
        question_embedding = self.generateEmbeddingsViaLangchain(question_text)
        question_embedding = np.array([question_embedding], dtype=np.float32)
        faiss.normalize_L2(question_embedding)

        # --- Step 4: Load FAISS Index + Metadata ---
        vecdb_path = os.path.join(self.vecdb_base_path, thread_id)
        index_path = os.path.join(vecdb_path, "index.faiss")
        metadata_path = os.path.join(vecdb_path, "metadata.json")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            print(f"Vector DB not found for thread ID: {thread_id}")
            return []

        index = faiss.read_index(index_path)
        distances, indices = index.search(question_embedding, index.ntotal)

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # --- Step 5: Filter and Prepare Results ---
        filtered_results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            vector_id = str(idx)
            result_metadata = metadata.get(vector_id)
            if not result_metadata:
                continue

            window_id = result_metadata.get("window_id")
            if window_id not in matching_window_ids:
                continue

            result_metadata["similarity_score"] = float(distances[0][i])
            result_metadata["vector_id"] = vector_id
            filtered_results.append(result_metadata)

        # Sort by similarity score (cosine similarity — higher is better)
        filtered_results.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Return top-k results
        print(f"Returning top {min(top_k, len(filtered_results))} results")
        return filtered_results[:top_k]
    
    def search_by_text(self, search_text, limit=10):
        """Search context windows for text in query or answer"""
        matching_windows = []
        
        # Get all keys that are context windows
        window_keys = self.redis_client.keys("window:*")
        
        for key in window_keys:
            window_data = self.redis_client.hgetall(key)
            
            # Check if search text is in the query or answer text
            query_text = window_data.get("query_text", "").lower()
            answer_text = window_data.get("answer_text", "").lower()
            
            if search_text.lower() in query_text or search_text.lower() in answer_text:
                matching_windows.append(window_data)
        
        return matching_windows[:limit]