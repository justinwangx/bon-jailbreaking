import streamlit as st
import json
from pathlib import Path
import glob
import csv
from datetime import datetime

class JailbreakVerifier:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.jailbreak_data = []
        self.fp_log_file = self.base_dir / "false_positives.csv"
        self.load_jailbreaks()
    
    def load_jailbreaks(self):
        for n in range(159):
            prompts_dir = self.base_dir / str(n) / "prompts"
            jailbreaks_dir = prompts_dir / "jailbreaks"
            
            if jailbreaks_dir.exists():
                jailbreak_files = glob.glob(str(jailbreaks_dir / "jailbreak_*_*.txt"))
                for jb_file in jailbreak_files:
                    a, b = map(int, jb_file.split('jailbreak_')[1].split('.txt')[0].split('_'))
                    response_file = prompts_dir / str(a) / str(b) / "classifier_responses_1.json"
                    if response_file.exists():
                        self.jailbreak_data.append({
                            'behavior_id': n,
                            'a': a,
                            'b': b,
                            'response_file': response_file,
                            'data': json.loads(response_file.read_text())[0]
                        })
    
    def log_false_positive(self, behavior_id, a, b):
        file_exists = self.fp_log_file.exists()
        with open(self.fp_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['behavior_id', 'a', 'b', 'timestamp'])
            writer.writerow([behavior_id, a, b, datetime.now().isoformat()])

def save_classification(response_file, data, new_value):
    data['classifier_outputs'][0] = new_value
    with open(response_file, 'w') as f:
        json.dump([data], f)

def main():
    st.title("Jailbreak Response Verifier")
    
    verifier = JailbreakVerifier("exp/bon/text/Llama-3-8B-Instruct-RR")
    
    # Session state for current index
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("Previous") and st.session_state.current_index > 0:
            st.session_state.current_index -= 1
    with col3:
        if st.button("Next") and st.session_state.current_index < len(verifier.jailbreak_data) - 1:
            st.session_state.current_index += 1
    
    # Statistics
    total_jailbreaks = len(verifier.jailbreak_data)
    false_positives = sum(1 for item in verifier.jailbreak_data 
                         if item['data']['classifier_outputs'][0] == "No")
    
    if total_jailbreaks > 0:
        fp_rate = (false_positives / total_jailbreaks) * 100
        st.sidebar.markdown(f"""
        ### Statistics
        - Total Jailbreaks: {total_jailbreaks}
        - False Positives: {false_positives}
        - FP Rate: {fp_rate:.1f}%
        """)
    
    # Current item
    if verifier.jailbreak_data:
        item = verifier.jailbreak_data[st.session_state.current_index]
        
        # Progress indicator with file information
        st.markdown(f"""
        **Progress:** {st.session_state.current_index + 1} of {len(verifier.jailbreak_data)}  
        **Behavior ID:** {item['behavior_id']}  
        **File Location:** prompts/{item['a']}/{item['b']}/classifier_responses_1.json
        """)
        
        st.subheader("Behavior:")
        st.text_area("", item['data']['behavior_str'], height=100, key="behavior", disabled=True)
        
        current_class = item['data']['classifier_outputs'][0]
        st.write(f"Current Classification: {current_class}")
        
        st.subheader("Response:")
        st.text_area("", item['data']['response'], height=300, key="response", disabled=True)
        
        if st.button("Toggle Classification"):
            new_value = "No" if current_class == "Yes" else "Yes"
            save_classification(item['response_file'], item['data'], new_value)
            # Log false positive when toggling to "No"
            if new_value == "No":
                verifier.log_false_positive(item['behavior_id'], item['a'], item['b'])
            # Advance to next item if not at the end
            if st.session_state.current_index < len(verifier.jailbreak_data) - 1:
                st.session_state.current_index += 1
            st.rerun()
        
        # Direct navigation at bottom
        st.markdown("---")
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            new_index = st.number_input(
                "Go to index:",
                min_value=1,
                max_value=len(verifier.jailbreak_data),
                value=st.session_state.current_index + 1,
                step=1
            )
            if new_index != st.session_state.current_index + 1:
                st.session_state.current_index = new_index - 1
                st.rerun()

if __name__ == "__main__":
    main()