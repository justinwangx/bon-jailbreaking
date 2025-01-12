import os
import json
from pathlib import Path

def find_jailbreak_pairs(base_dir):
    results = []
    
    # Iterate through all numbered directories (0-158)
    for dir_num in range(159):
        dir_path = Path(base_dir) / str(dir_num)
        jailbreak_dir = dir_path / 'prompts' / 'jailbreaks'
        
        # Skip if jailbreak directory doesn't exist
        if not jailbreak_dir.exists():
            continue
            
        # Find all jailbreak files in the directory
        for jb_file in jailbreak_dir.glob('jailbreak_*_*.txt'):
            # Extract the two numbers from the filename
            nums = [int(n) for n in jb_file.stem.split('_')[1:]]
            
            # Construct paths to the corresponding response files
            response_dir = dir_path / 'prompts' / str(nums[0]) / str(nums[1])
            lm_response = response_dir / 'lm_responses_1.json'
            classifier_response = response_dir / 'classifier_responses_1.json'
            
            if lm_response.exists() and classifier_response.exists():
                results.append({
                    'directory': str(dir_num),
                    'jailbreak_file': str(jb_file),
                    'lm_response': str(lm_response),
                    'classifier_response': str(classifier_response)
                })
    
    return results

def analyze_jailbreak_responses(base_dir):
    pairs = find_jailbreak_pairs(base_dir)
    results = []
    
    for pair in pairs:
        # Read jailbreak content
        with open(pair['jailbreak_file'], 'r') as f:
            jb_content = f.read().strip()
        
        # Read and parse response files
        with open(pair['lm_response'], 'r') as f:
            lm_data = json.load(f)
        with open(pair['classifier_response'], 'r') as f:
            classifier_data = json.load(f)
            
        # Store all data in a dictionary
        result = {
            'directory': pair['directory'],
            'jailbreak_file': pair['jailbreak_file'],
            'jailbreak_prompt': jb_content,
            'lm_response': lm_data[0]['completion'].strip(),
            'classifier_response': classifier_data
        }
        results.append(result)
        
        # Optional: Still print progress to console
        print(f"Processed directory {pair['directory']}")
    
    # Save all results to a JSON file
    output_file = Path(base_dir) / 'jailbreak_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis complete. Results saved to {output_file}")
    print(f"Found {len(results)} jailbreak attempts")

if __name__ == "__main__":
    base_directory = "exp/bon/text/Llama-3-8B-Instruct-RR"
    analyze_jailbreak_responses(base_directory)