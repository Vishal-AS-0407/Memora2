# demo.py

from memory_system import MemoraBrain
import os

def print_separator():
    print("\n" + "="*50 + "\n")

def main():
    # Initialize the memory system
    memory = MemoraBrain(vecdb_base_path="./vecdb_storage")
    
    print("🧠 Memora - LLM Memory System Demo")
    print_separator()
    
    # Create some initial context windows
    print("📝 Creating initial conversations...")
    
    # First conversation about neural networks
    nn_result = memory.create_context_window(
        query_text="How do neural networks work?",
        answer_text="Neural networks are computational systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process and transmit information. These systems learn from data through a process called training, where they adjust the connections between neurons to minimize error.",
        user_name="Alice"
    )
    
    # Second conversation about transfer learning (referring to the first one)
    tl_result = memory.create_context_window(
        query_text="What is transfer learning?",
        answer_text="Transfer learning is a technique where a model developed for a task is reused as the starting point for a model on a second task. This is particularly useful in deep learning where pre-trained models are used as feature extractors or fine-tuned for specific applications.",
        referral_ids=[nn_result["window_id"]],
        user_name="Alice"
    )
    
    # Third conversation by a different user
    rl_result = memory.create_context_window(
        query_text="Explain reinforcement learning",
        answer_text="Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize some notion of cumulative reward. It differs from supervised learning in that the agent is not provided with correct input/output pairs but must discover them through a process of trial and error.",
        user_name="Bob"
    )
    
    # Let's print out what we've stored
    print(f"Created window: {nn_result['window_id']} (Topics: {nn_result['topics']})")
    print(f"Created window: {tl_result['window_id']} (Topics: {tl_result['topics']})")
    print(f"Created window: {rl_result['window_id']} (Topics: {rl_result['topics']})")
    print_separator()
    
    # Let's retrieve a specific context window
    print("📂 Retrieving a specific context window...")
    window = memory.get_context_window(nn_result["window_id"])
    print(f"Window ID: {window['window_id']}")
    print(f"Query: {window['query_text']}")
    print(f"Answer: {window['answer_text']}")
    print(f"User: {window['user_name']}")
    print(f"Topics: {window['topics']}")
    print_separator()
    
    # Let's search by text
    print("🔍 Searching by text...")
    results = memory.search_by_text("neural")
    print(f"Found {len(results)} results containing 'neural':")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['query_text'][:50]}... by {result['user_name']}")
    print_separator()
    
    # Let's search by user
    print("👤 Searching by user...")
    results = memory.search_by_user("Alice")
    print(f"Found {len(results)} conversations by Alice:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['query_text'][:50]}...")
    print_separator()
    
    # Let's do a semantic search
    print("🔍 Performing semantic search...")
    query = "How do machine learning models transfer knowledge?"
    results = memory.semantic_search(query)
    print(f"Semantic search for: '{query}'")
    print(f"Found {len(results)} relevant results:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['question'][:50]}... (distance: {result['distance']:.4f})")
    print_separator()
    
    # Let's look at the thread structure
    print("🧵 Exploring thread structure...")
    thread_id = nn_result["thread_id"]
    thread_index = memory.get_thread_index(thread_id)
    print(f"Thread ID: {thread_id}")
    print("Topics in this thread:")
    for topic, window_ids in thread_index.items():
        if topic != "all_window_keys":
            print(f"- {topic}: {len(window_ids)} windows")
    print_separator()
    
    print("✅ Demo completed!")

if __name__ == "__main__":
    main()