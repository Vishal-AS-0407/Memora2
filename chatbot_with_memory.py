# chatbot_with_memory.py

from memory_system import MemoraBrain
import os
import json
from datetime import datetime

class MemoraChatbot:
    def __init__(self, username="DefaultUser"):
        self.memory = MemoraBrain(vecdb_base_path="./vecdb_storage")
        self.username = username
        self.current_thread_id = None
        
    def start_new_conversation(self):
        """Start a new conversation thread"""
        self.current_thread_id = None
        print(f"Starting a new conversation with {self.username}...")
        
    def continue_conversation(self, thread_id):
        """Continue an existing conversation thread"""
        if self.memory.get_thread_index(thread_id):
            self.current_thread_id = thread_id
            print(f"Continuing conversation in thread {thread_id}...")
            return True
        else:
            print(f"Thread {thread_id} not found. Starting a new conversation...")
            return False
            
    def process_query(self, query, mock_llm_response=None):
        """Process a user query, with memory integration"""
        print(f"\n👤 {self.username}: {query}")
        
        # First, search for relevant context from memory
        relevant_context = []
        
        # If we're in a thread, first check that thread
        if self.current_thread_id:
            thread_results = self.memory.semantic_search(query, thread_id=self.current_thread_id, top_k=2)
            relevant_context.extend(thread_results)
        
        # Always do a global search too
        global_results = self.memory.semantic_search(query, top_k=3)
        
        # Combine results, removing duplicates
        existing_ids = {r["vector_id"] for r in relevant_context}
        for result in global_results:
            if result["vector_id"] not in existing_ids:
                relevant_context.append(result)
                existing_ids.add(result["vector_id"])
        
        # Format context for augmenting the LLM prompt
        formatted_context = []
        referral_ids = []
        
        for i, ctx in enumerate(relevant_context[:3]):  # Use top 3 most relevant
            formatted_context.append(f"Context {i+1}:\nQ: {ctx['question']}\nA: {ctx['answer']}")
            if 'window_id' in ctx:
                referral_ids.append(ctx['window_id'])
        
        # In a real system, you would send this context to your LLM
        context_for_llm = "\n\n".join(formatted_context)
        
        # For demo purposes, we'll use a mock LLM response if provided
        if mock_llm_response:
            answer = mock_llm_response
        else:
            # In a real system, this would be the LLM call
            # response = llm.generate(query, context=context_for_llm)
            answer = f"This is a simulated response to: '{query}'\nBased on context from {len(relevant_context)} previous interactions."
        
        print(f"🤖 Memora: {answer}")
        
        # Store this interaction in memory
        result = self.memory.create_context_window(
            query_text=query,
            answer_text=answer,
            referral_ids=referral_ids if referral_ids else None,
            user_name=self.username
        )
        
        # Update current thread ID if we didn't have one
        if not self.current_thread_id:
            self.current_thread_id = result["thread_id"]
        
        return {
            "answer": answer,
            "window_id": result["window_id"],
            "thread_id": result["thread_id"],
            "topics": result["topics"]
        }

    def get_conversation_history(self):
        """Get the full conversation history for the current thread"""
        if not self.current_thread_id:
            return []
            
        thread_index = self.memory.get_thread_index(self.current_thread_id)
        if not thread_index or "all_window_keys" not in thread_index:
            return []
            
        window_ids = thread_index["all_window_keys"]
        history = []
        
        for window_id in window_ids:
            window = self.memory.get_context_window(window_id)
            if window:
                history.append({
                    "role": "user" if window["user_name"] else "system",
                    "content": window["query_text"],
                    "timestamp": datetime.fromtimestamp(window["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                })
                history.append({
                    "role": "assistant",
                    "content": window["answer_text"],
                    "timestamp": datetime.fromtimestamp(window["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                })
                
        return history

def demo():
    print("🤖 Memora Chatbot with Memory")
    print("=" * 50)
    
    # Initialize chatbot with a username
    username = input("Please enter your name: ").strip() or "User"
    chatbot = MemoraChatbot(username=username)
    
    # Start a new conversation
    chatbot.start_new_conversation()
    
    # Some example interactions with mock responses for demo purposes
    mock_responses = {
        "Tell me about machine learning": 
            "Machine learning is a field of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions based on those patterns.",
        
        "How does neural network training work?":
            "Neural network training works through a process called backpropagation. First, data is fed forward through the network to generate predictions. Then, the error between predictions and actual values is calculated. This error is propagated backward through the network to adjust the weights of connections between neurons, gradually improving the network's performance.",
        
        "What's the difference between supervised and unsupervised learning?":
            "Supervised learning uses labeled data, where the model learns to map inputs to known outputs. Unsupervised learning works with unlabeled data, where the model identifies patterns and structures in the data without specific guidance. Supervised learning is used for tasks like classification and regression, while unsupervised learning is used for clustering, dimensionality reduction, and anomaly detection."
    }
    
    # Simulate a conversation
    for query, response in mock_responses.items():
        result = chatbot.process_query(query, mock_llm_response=response)
        print(f"Topics detected: {', '.join(result['topics'])}")
        print("-" * 50)
    
    # Show conversation history
    print("\n📜 Conversation History:")
    history = chatbot.get_conversation_history()
    for message in history:
        role_symbol = "👤" if message["role"] == "user" else "🤖"
        print(f"[{message['timestamp']}] {role_symbol} {message['role']}: {message['content'][:50]}...")
    
    # Interactive mode
    print("\n💬 Now you can chat! (type 'exit' to quit)")
    while True:
        user_input = input(f"\n👤 {username}: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
            
        result = chatbot.process_query(user_input)
        print(f"Topics detected: {', '.join(result['topics'])}")

if __name__ == "__main__":
    demo()