import json
import argparse
import os
import shutil

def copy_audio_files(input_file, output_dir, output_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    updated_data = []
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            audio_file = data['audio_file'].replace('.mp3', '.wav')
            new_audio_file_path = os.path.join(output_dir, os.path.basename(audio_file))
            shutil.copy(audio_file, new_audio_file_path)
            data['audio_file'] = new_audio_file_path
            updated_data.append(data)
    
    with open(output_file, 'w') as f:
        for data in updated_data:
            f.write(json.dumps(data) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy audio files to output directory.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input jsonl file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output jsonl file.')
    args = parser.parse_args()

    copy_audio_files(args.input_file, args.output_dir, args.output_file)
