import json
import os
from typing import Dict
from pathlib import Path

import torchaudio
from tqdm import tqdm

def load_transcripts(transcripts_txt_path):
    transcripts:Dict[str, str] = {}
    with open(transcripts_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip()
            if len(parts) == 0:
                continue
            parts = line.split()
            transcripts[parts[0]] = ''.join(parts[1:])
    return transcripts

def get_duration_distribution(paths):
    durations = []
    total_duration = 0.0
    buckets_duration = 0.5  # seconds
    for path in tqdm(paths):
        info = torchaudio.info(path)
        duration = info.num_frames / info.sample_rate
        durations.append(duration)
        total_duration += duration
    print(f"Total duration: {total_duration/3600:.2f} hours")
    print(f"Average duration: {total_duration / len(paths)}")
    buckets = {}
    for duration in (durations):
        bucket = int(duration // buckets_duration)
        buckets[bucket] = buckets.get(bucket, 0) + 1

    print(f"Duration distribution (bucket size: {buckets_duration}s):")
    for bucket, count in sorted(buckets.items()):
        print(f"{bucket * buckets_duration:.1f}s - {(bucket + 1) * buckets_duration:.1f}s: {count}")

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    transcripts_txt_path = f'{root}/dataset/aishell1/data_aishell/transcript/aishell_transcript_v0.8.txt'
    transcripts = load_transcripts(transcripts_txt_path)
    print(f"Loaded {len(transcripts)} transcripts.")
    for split in ['train', 'dev', 'test']:
        audio_dir = f'{root}/dataset/aishell1/data_aishell/wav/{split}'
        output_json_path = f'{root}/dataset/aishell1/data_aishell/transcript/aishell1_{split}.jsonl'
        audio_files = sorted(Path(audio_dir).rglob('*.wav'))
        with open(output_json_path, 'w', encoding='utf-8') as json_f:
            for audio_file in audio_files:
                file_id = audio_file.stem
                if file_id in transcripts:
                    info = torchaudio.info(str(audio_file))
                    duration = info.num_frames / info.sample_rate
                    data = {
                        "path": str(audio_file),
                        "txt": transcripts[file_id],
                        "key": file_id,
                        "duration": duration
                    }
                json_f.write(json.dumps(data, ensure_ascii=False) + '\n')
        print(f'get duration distribution for {split} set:')
        get_duration_distribution(audio_files)

if __name__ == "__main__":
    main()