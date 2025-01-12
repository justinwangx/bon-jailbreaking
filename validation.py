import os
from pathlib import Path
import glob

def count_jailbreaks(base_dir):
    total_jailbreaks = 0
    behavior_dirs = {}  # To store counts per behavior
    
    # Iterate through behavior directories (0-158)
    for n in range(159):
        prompts_dir = Path(base_dir) / str(n) / "prompts"
        jailbreaks_dir = prompts_dir / "jailbreaks"
        
        if jailbreaks_dir.exists():
            # Find all jailbreak files
            jailbreak_files = glob.glob(str(jailbreaks_dir / "jailbreak_*_*.txt"))
            count = len(jailbreak_files)
            if count > 0:
                behavior_dirs[n] = count
                total_jailbreaks += count
    
    return total_jailbreaks, behavior_dirs

if __name__ == "__main__":
    base_dir = "exp/bon/text/Llama-3-8B-Instruct-RR"
    total, by_behavior = count_jailbreaks(base_dir)
    
    print(f"Total jailbreaks to verify: {total}")
    print("\nBreakdown by behavior:")
    for behavior_id, count in sorted(by_behavior.items()):
        print(f"Behavior {behavior_id}: {count} jailbreaks")