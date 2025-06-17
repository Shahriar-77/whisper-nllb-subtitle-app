from pathlib import Path
import pandas as pd
import ffmpeg
import os
from faster_whisper import WhisperModel
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
import re
from datetime import timedelta
from typing import List, Dict
from translator import Translator
from tqdm import tqdm


# Normalize file names
def normalize_filename(path: Path) -> Path:
    clean_name = re.sub(r"[^\w\s-]", "", path.stem)  # Remove special characters
    clean_name = clean_name.replace(" ", "_").replace("-", "_")
    new_path = path.with_name(clean_name + path.suffix)

    if new_path != path:
        path.rename(new_path)

    return new_path

# Centralized list of convertible suffixes
CONVERTIBLE_SUFFIXES = [".mp4", ".mkv", ".avi", ".mp3", ".flac",".m4a",".ts"]
VALID_SUFFIXES = CONVERTIBLE_SUFFIXES + [".wav"]

# Scan Input Directory and make a DataFrame
def collect_input_files(input_dir, valid_suffixes=None):
    valid_suffixes = valid_suffixes or VALID_SUFFIXES
    input_dir = Path(input_dir)
    
    files = [f for f in input_dir.rglob("*") if f.suffix.lower() in valid_suffixes]
    
    renamed_files = []
    for f in files:
        new_path = normalize_filename(f)
        renamed_files.append(new_path)


    df = pd.DataFrame([{
        "path": f,
        "name": f.stem,
        "suffix": f.suffix.lower(),
        "needs_conversion": f.suffix.lower() in CONVERTIBLE_SUFFIXES,
    } for f in renamed_files])
    
    return df


# Convert Audio and Video to Whipser-Compatible .wav
def convert_to_whisper_wav(input_path: Path, output_dir: Path):
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (input_path.stem + ".wav")

    (
        ffmpeg
        .input(str(input_path))
        .output(str(output_path), ac=1, ar=16000, format='wav', vn=None)
        .overwrite_output()
        .run(quiet=True)
    )
    return output_path


# Pre-process all files
def preprocess_all(df: pd.DataFrame, output_audio_dir: Path):
    
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    
    df["processed_path"] = df.apply(
        lambda row: convert_to_whisper_wav(row["path"], output_audio_dir)
        if row["needs_conversion"] else row["path"], axis=1
    )
    return df


## Load Whipser model

def load_whisper_model(model_size="medium"):
    """Load the whisper mode with specified model size"""
    model = WhisperModel(
        model_size,
        device="cuda",
        compute_type="float16",  # Using float16 for better VRAM handling
    )
    return model

def convert_segments_to_dict(segments):
    """Convert Whisper Segment objects to plain dictionaries."""
    converted = []
    for seg in segments:
        seg_dict = {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text
        }
        if hasattr(seg, "words"):
            seg_dict["words"] = [
                {"start": w.start, "end": w.end, "word": w.word}
                for w in seg.words
                if hasattr(w, "start") and hasattr(w, "end") and hasattr(w, "word")
            ]
        converted.append(seg_dict)
    return converted


def transcribe_audio_file(model, audio_path, task="transcribe", beam_size=5, language=None):
    """Transcribe Audio File, also retrun word timestamps to aid subtitle generation."""
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=beam_size,
        task=task,
        language=language,
        word_timestamps=True,
    )
    
    
    segments = list(segments)
    segments = convert_segments_to_dict(segments)
    full_text = " ".join([seg["text"] for seg in segments])
    
    return full_text, segments, info






# Subtitle Creation
def format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp (hh:mm:ss,ms)."""
    if seconds < 0:
        seconds = 0
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)

    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def format_ass_timestamp(seconds: float) -> str:
    """Format seconds to ASS timestamp: h:mm:ss.cs (centiseconds)"""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int((seconds - int(seconds)) * 100)
    return f"{hours}:{minutes:02}:{secs:02}.{centis:02}"






def chunk_segment_by_words(segment: Dict, max_chunk_length=6.0) -> List[Dict]:
    """Split a long segment into smaller chunks using word timestamps."""
    if "words" not in segment:
        return [segment]  # Cannot chunk without word timestamps

    words = segment["words"]
    chunks = []
    current_chunk = []
    chunk_start = None

    for word in words:
        if word.get("start") is None or word.get("end") is None:
            continue  # Skip malformed words

        if chunk_start is None:
            chunk_start = word["start"]

        current_chunk.append(word)

        # If current chunk exceeds length OR end of list
        chunk_duration = word["end"] - chunk_start
        if chunk_duration >= max_chunk_length or word == words[-1]:
            chunks.append({
                "start": chunk_start,
                "end": word["end"],
                "text": " ".join(w["word"].strip() for w in current_chunk),
            })
            current_chunk = []
            chunk_start = None

    return chunks

def repair_segments_hybrid(segments: List[Dict], max_duration=7.0) -> List[Dict]:
    """Fix long segments using either timestamp realignment or word chunking."""
    repaired = []
    for i, seg in enumerate(segments):
        duration = seg["end"] - seg["start"]

        if duration <= max_duration:
            repaired.append(seg)
        else:
            # Try word-based chunking first
            if "words" in seg and seg["words"]:
                new_chunks = chunk_segment_by_words(seg, max_chunk_length=max_duration)
                repaired.extend(new_chunks)
            else:
                # Fallback: adjust start & end using surrounding segments
                prev_end = repaired[-1]["end"] + 0.5 if repaired else max(0.0, seg["start"])
                next_start = segments[i + 1]["start"] - 0.5 if i + 1 < len(segments) else seg["end"]

                repaired.append({
                    "start": prev_end,
                    "end": max(prev_end + 1.0, next_start),  # at least 1s duration
                    "text": seg["text"]
                })

    return repaired


def save_transcript(segments, output_path: Path, formats=['txt'],max_duration=7.0):
    """Save transcript in multiple formats."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    
    # Repair and validate segments before saving
    segments = repair_segments_hybrid(segments, max_duration=max_duration)

    

    if 'txt' in formats:
        with open(output_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
            for seg in segments:
                f.write(f"{seg['text'].strip()}\n")

    if 'srt' in formats:
        with open(output_path.with_suffix('.srt'), 'w', encoding='utf-8') as f:
            for i, seg in enumerate(segments, 1):
                
                start = max(0.0, float(seg["start"]))
                end = max(start + 0.001, float(seg["end"]))
                if end <= start:
                    print(f"⚠️ Invalid segment: {seg['text'][:40]} (start={start}, end={end})")
                    end = start + 0.5  # fallback
                
                f.write(f"{i}\n")
                f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
                f.write(f"{seg['text'].strip()}\n\n")

    if 'ass' in formats:
        with open(output_path.with_suffix('.ass'), 'w', encoding='utf-8') as f:
            f.write("[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\n")
            f.write("Format: Name, Fontname, Fontsize, PrimaryColour, Bold, Italic, Alignment, MarginL, MarginR, MarginV, Encoding\n")
            f.write("Style: Default,Arial,20,&H00FFFFFF,-1,0,2,10,10,10,1\n\n[Events]\n")
            f.write("Format: Marked, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

            for seg in segments:
                
                start = max(0.0, float(seg["start"]))
                end = max(start + 0.001, float(seg["end"]))
                
                if end <= start:
                    print(f"⚠️ Invalid segment: {seg['text'][:40]} (start={start}, end={end})")
                    end = start + 0.5  # fallback
                
                start_str = format_ass_timestamp(start)
                end_str = format_ass_timestamp(end)
                f.write(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{seg['text'].strip()}\n")
                
    if 'vtt' in formats:
        with open(output_path.with_suffix('.vtt'), 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for i, seg in enumerate(segments, 1):
                start = max(0.0, float(seg["start"]))
                end = max(start + 0.001, float(seg["end"]))
                if end <= start:
                    end = start + 0.5
                start_str = format_timestamp(start).replace(",", ".")
                end_str = format_timestamp(end).replace(",", ".")
                f.write(f"{start_str} --> {end_str}\n{seg['text'].strip()}\n\n")            


# Other lang subtitle creation from original .srt/.ass file 
def translate_srt_file(srt_path, translator: Translator, src_lang="eng_Latn", tgt_lang="ben_Beng", save_path=None):
    output_lines = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    block = []
    for line in lines + ["\n"]:  # Ensure trailing newline to flush last block
        if line.strip() == "":
            if len(block) >= 3:
                # block[0]: sequence number
                # block[1]: timestamp
                # block[2:]: text
                text_lines = block[2:]
                original_text = " ".join(line.strip() for line in text_lines)
                translated_text = translator.translate(original_text, src_lang=src_lang, tgt_lang=tgt_lang)
                translated_lines = [translated_text] if len(text_lines) == 1 else translated_text.split("\n")
                output_lines.extend([block[0], block[1]] + translated_lines + ["\n"])
            else:
                output_lines.extend(block + ["\n"])
            block = []
        else:
            block.append(line.strip())

    if save_path is None:
        save_path = Path(srt_path).with_name(f"{Path(srt_path).stem}_{tgt_lang}.srt")

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))

    print(f"Translated SRT saved to: {save_path}")

def translate_ass_file(ass_path, translator: Translator,src_lang="eng_Latn", tgt_lang="ben_Beng", save_path=None):
    output_lines = []
    in_events = False

    with open(ass_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().lower().startswith("[events]"):
                in_events = True
                output_lines.append(line)
                continue
            if in_events and line.startswith("Dialogue:"):
                parts = line.split(",", 9)
                if len(parts) >= 10:
                    original_text = parts[9].strip()
                    translated_text = translator.translate(original_text, src_lang=src_lang, tgt_lang=tgt_lang)
                    parts[9] = translated_text
                    output_lines.append(",".join(parts) + "\n")
                else:
                    output_lines.append(line)
            else:
                output_lines.append(line)

    if save_path is None:
        save_path = Path(ass_path).with_name(f"{Path(ass_path).stem}_{tgt_lang}.ass")

    with open(save_path, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)

    print(f"Translated ASS saved to: {save_path}")


def translate_vtt_file(vtt_path, translator: Translator, src_lang="eng_Latn", tgt_lang="ben_Beng", save_path=None):
    output_lines = ["WEBVTT\n"]
    with open(vtt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    block = []
    for line in lines[1:] + ["\n"]:  # Skip header and ensure last block flushes
        if line.strip() == "":
            if len(block) >= 2 and "-->" in block[0]:
                timestamp_line = block[0]
                text_lines = block[1:]
                original_text = " ".join(line.strip() for line in text_lines)
                translated_text = translator.translate(original_text, src_lang=src_lang, tgt_lang=tgt_lang)
                translated_lines = [translated_text] if len(text_lines) == 1 else translated_text.split("\n")
                output_lines.append(timestamp_line)
                output_lines.extend([line.strip() for line in translated_lines])
                output_lines.append("")  # blank line between entries
            else:
                output_lines.extend(block + [""])
            block = []
        else:
            block.append(line.strip())

    if save_path is None:
        save_path = Path(vtt_path).with_name(f"{Path(vtt_path).stem}_{tgt_lang}.vtt")

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))

    print(f"Translated VTT saved to: {save_path}")



