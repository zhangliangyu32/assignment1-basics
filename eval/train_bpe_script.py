import argparse
import cProfile
import pickle
from cs336_basics.Tokenizer import BPETrainer
import time
def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on a dataset")
    parser.add_argument("input_path", type=str, help="Path to the input text file")
    parser.add_argument("vocab_size", type=int, help="Size of the vocabulary")
    parser.add_argument("--special-tokens", nargs="+", default=["<|endoftext|>"], 
                       help="List of special tokens (default: ['<|endoftext|>'])")
    parser.add_argument("--vocab-output", type=str, default="./vocabulary.pkl",
                       help="Output path for vocabulary file (default: ./vocabulary.pkl)")
    parser.add_argument("--merges-output", type=str, default="./merges.pkl",
                       help="Output path for merges file (default: ./merges.pkl)")
    parser.add_argument("--profile", action="store_true", default=False, 
                       help="Run with cProfile for performance profiling")
    
    args = parser.parse_args()
    trainer = BPETrainer()
    def run_training():
            return trainer.train(args.input_path, args.vocab_size, args.special_tokens)
    if args.profile:
        # Use cProfile to profile the training
        
        print("Starting profiled training...")
        profiler = cProfile.Profile()
        profiler.enable()
        time_start = time.time()
        vocabulary, merges = run_training()
        time_ed = time.time()
        print("time taken to train the tokenizer", time_ed - time_start)
        profiler.disable()
        
        # Print the profile stats
        print("\n=== PROFILING RESULTS (TOP 10 BY TOTAL TIME) ===")
        profiler.print_stats(sort='tottime')
        print("\n=== PROFILING RESULTS (TOP 10 BY CUMULATIVE TIME) ===")
        profiler.print_stats(sort='cumulative')
    else:
        # Run normally without profiling
        time_start = time.time()
        vocabulary, merges = run_training()
        time_ed = time.time()
        print("time taken to train the tokenizer", time_ed - time_start)
    # Save the results
    pickle.dump(vocabulary, open(args.vocab_output, "wb"))
    pickle.dump(merges, open(args.merges_output, "wb"))
    
    print(f"Training completed!")
    print(f"Vocabulary saved to: {args.vocab_output}")
    print(f"Merges saved to: {args.merges_output}")
    print(f"Vocabulary size: {len(vocabulary)}")
    max_len = 1
    max_len_token = None
    for id, token in vocabulary.items():
        if len(token) > max_len:
            max_len = len(token)
            max_len_token = token
    print(f"Max length token: {max_len_token}, Length: {max_len}")


if __name__ == "__main__":
    main()