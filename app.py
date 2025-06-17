import streamlit as st
from pathlib import Path
import pandas as pd
import os
import torch 

# Prevent "__path__._path" error in some Streamlit + torch.jit setups
try:
    torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
except Exception:
    pass  # Safe fallback if already resolved or not needed

from main import (
    collect_input_files,
    preprocess_all,
    load_whisper_model,
    transcribe_audio_file,
    save_transcript,
    translate_srt_file,
    translate_ass_file,
    translate_vtt_file
)
from translator import Translator

# --- Title ---
st.set_page_config(page_title="Audio Transcriber & Translator", layout="wide")
st.title("🎧 Audio/Video Transcription & Translation App")

# --- Init Session State ---
if "current_step" not in st.session_state:
    st.session_state.current_step = None
if "stop_flag" not in st.session_state:
    st.session_state.stop_flag = False

# --- Step Helper ---
def step_button(label, key, allowed_states, set_to=None):
    if st.session_state.current_step not in allowed_states:
        return False
    clicked = st.button(label, key=key)
    if clicked and set_to:
        st.session_state.current_step = set_to
        
        
    return clicked


# --- Sidebar ---
st.sidebar.header("1️⃣ Input Settings")
input_dir = st.sidebar.text_input("Input Folder", value="samples/")
output_audio_dir = st.sidebar.text_input("Preprocessed Audio Output Folder", value="outputs/audio_wav/")
output_subtitle_dir = st.sidebar.text_input("Subtitle Output Folder", value="outputs/subtitles/")

task = st.sidebar.radio("Task", ["Transcribe", "Translate"])
model_size = st.sidebar.selectbox("Whisper Model Size", ["tiny", "base", "small", "medium", "large"], index=3)

# Whisper: High-accuracy source language selection
HIGH_ACCURACY_LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "Italian": "it", "German": "de",
    "Portuguese": "pt", "Dutch": "nl", "Swedish": "sv", "Norwegian": "no", "Danish": "da",
    "Finnish": "fi", "Polish": "pl", "Romanian": "ro", "Catalan": "ca", "Czech": "cs",
    "Turkish": "tr", "Hungarian": "hu", "Indonesian": "id", "Malay": "ms", "Japanese": "ja",
    "Korean": "ko", "Chinese": "zh", "Russian": "ru", "Ukrainian": "uk", "Greek": "el"
}

# All WHISPER languages 
WHISPER_LANGUAGES = {
    "Afrikaans": "af", "Albanian": "sq", "Amharic": "am", "Arabic": "ar", "Armenian": "hy",
    "Assamese": "as", "Azerbaijani": "az", "Basque": "eu", "Belarusian": "be", "Bengali": "bn",
    "Bosnian": "bs", "Breton": "br", "Bulgarian": "bg", "Burmese": "my", "Catalan": "ca",
    "Chinese": "zh", "Croatian": "hr", "Czech": "cs", "Danish": "da", "Dutch": "nl",
    "English": "en", "Estonian": "et", "Faroese": "fo", "Finnish": "fi", "French": "fr",
    "Galician": "gl", "Georgian": "ka", "German": "de", "Greek": "el", "Gujarati": "gu",
    "Hausa": "ha", "Hebrew": "he", "Hindi": "hi", "Hungarian": "hu", "Icelandic": "is",
    "Indonesian": "id", "Italian": "it", "Japanese": "ja", "Javanese": "jv", "Kannada": "kn",
    "Kazakh": "kk", "Khmer": "km", "Korean": "ko", "Lao": "lo", "Latin": "la", "Latvian": "lv",
    "Lingala": "ln", "Lithuanian": "lt", "Macedonian": "mk", "Malay": "ms", "Malayalam": "ml",
    "Marathi": "mr", "Mongolian": "mn", "Nepali": "ne", "Norwegian": "no", "Nyanja": "ny",
    "Pashto": "ps", "Persian": "fa", "Polish": "pl", "Portuguese": "pt", "Punjabi": "pa",
    "Romanian": "ro", "Russian": "ru", "Sanskrit": "sa", "Serbian": "sr", "Shona": "sn",
    "Sindhi": "sd", "Sinhala": "si", "Slovak": "sk", "Slovenian": "sl", "Somali": "so",
    "Spanish": "es", "Sundanese": "su", "Swahili": "sw", "Swedish": "sv", "Tagalog": "tl",
    "Tamil": "ta", "Telugu": "te", "Thai": "th", "Turkish": "tr", "Ukrainian": "uk",
    "Urdu": "ur", "Uzbek": "uz", "Vietnamese": "vi", "Welsh": "cy", "Xhosa": "xh",
    "Yoruba": "yo", "Zulu": "zu"
}

# Give access to all Whisper languages if Users want
unlock_all_langs = st.sidebar.checkbox(
    "🔓 Unlock all Whisper languages (⚠️ may reduce transcription accuracy)",
    help="Enable this only if you need support for languages not in the default high-accuracy list. Transcription quality may vary."
)

available_languages = WHISPER_LANGUAGES if unlock_all_langs else HIGH_ACCURACY_LANGUAGES
lang_display = list(available_languages.keys())
selected_lang_name = st.sidebar.selectbox("Source Language (Whisper)", lang_display, index=0)
source_lang = available_languages[selected_lang_name]

st.sidebar.caption("🧠 Only languages with a Word Error Rate (WER) under 20% are listed to ensure high-quality transcription.")



# ============================
# 🌐 Task: NLLB Translation
# ============================
st.sidebar.header("🌐 Task: Subtitle Translation (NLLB-200)")

st.sidebar.markdown("🔗 [View supported NLLB-200 language codes](https://dl-translate.readthedocs.io/en/latest/available_languages/#nllb-200)")

src_lang = st.sidebar.text_input("Source Language Code (e.g., `eng_Latn`, `jpn_Jpan`, `ind_Latn`,`zsm_Latn` etc.)", value="eng_Latn")
tgt_lang = st.sidebar.text_input("Target Language Code (e.g., `ben_Beng`, `spa_Latn`, `fra_Latn`, etc.)", value="ben_Beng")

st.sidebar.caption("⚠️ NLLB translation requires manual language selection. Ensure your source language matches the subtitle file.")


st.info(f"📌 Current Step: `{st.session_state.current_step or 'idle'}`")


# --- Step 1: Scan Directory ---

st.header("🔍 Step 1: Scan Input Directory")

valid_suffixes = [".mp4", ".mkv", ".avi", ".mp3", ".wav", ".flac", ".m4a",".ts"]
input_path = Path(input_dir)

if step_button("Scan Directory", key="scan_btn", allowed_states=[None, "transcription_completed","translation_completed","batch_transcription_completed"], set_to="scanning"):
    st.success("✅ Scan step triggered.")

    if not input_path.exists() or not input_path.is_dir():
        st.error("❌ The specified input directory does not exist or is not a folder.")
        
    else:
        matched_files = [f for f in input_path.glob("**/*") if f.suffix.lower() in valid_suffixes]
        if not matched_files:
            st.error("⚠️ No supported audio/video files found in the selected folder.")
            
        else:
            df = collect_input_files(input_dir)
            st.session_state["file_df"] = df
            st.success(f"✅ Found {len(df)} valid media file(s).")
            st.session_state.current_step = "scanned"
            


# --- Step 2: Preprocess Files ---
st.header("⚙️ Step 2: Preprocess Files")
if "file_df" in st.session_state:
    if step_button("Preprocess Audio/Video", key="preprocess_btn", allowed_states=["scanned"], set_to="preprocessing"):
        st.success("✅ Preprocessing step triggered.")
        df_processed = preprocess_all(st.session_state["file_df"], Path(output_audio_dir))
        st.session_state["processed_df"] = df_processed
        st.success("✅ Preprocessing complete.")
        st.session_state.current_step="preprocessed"
        

# --- Step 3: Load Whisper Model ---
st.header("🤖 Step 3: Load Whisper Model")

# two accepted states one is after initial model loading, the other is for the case if transcription is not upto par then, to change model
if step_button("Load Whisper", key="load_model_btn", allowed_states=["preprocessed", "transcription_completed"], set_to="model_loaded"):
    st.success("✅ Model load step triggered.")
    whisper_model = load_whisper_model(model_size=model_size)
    st.session_state["whisper_model"] = whisper_model
    st.success(f"✅ Whisper model '{model_size}' loaded on GPU.")
    

# --- Step 4: Choose File and Transcribe ---
if "processed_df" in st.session_state and "whisper_model" in st.session_state:
    st.header("📝 Step 4: Transcribe / Translate")

    file_list = st.session_state["processed_df"]["processed_path"].tolist()
    selected_file = st.selectbox("Choose File to Transcribe/Translate", file_list)

    output_base = Path(output_subtitle_dir) / Path(selected_file).stem
    expected_formats = ["txt", "srt", "ass", "vtt"]
    existing_outputs = [f for f in expected_formats if (output_base.with_suffix(f".{f}")).exists()]

    if len(existing_outputs) == len(expected_formats):
        st.info("✅ This file has already been transcribed/translated.")

        for fmt in existing_outputs:
            file_path = output_base.with_suffix(f".{fmt}")
            st.markdown(f"**Preview: `{file_path.name}`**")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                    preview = "\n".join(lines[:4]) if fmt != "txt" else "\n".join(lines[:2])
                    st.code(preview, language="text")
            except Exception as e:
                st.error(f"⚠️ Could not read {fmt} file: {e}")

        force_reprocess = st.checkbox("🔁 Reprocess Anyway?")
        if force_reprocess:
            st.warning("⚠️ Only reprocess if you changed model size, source language, or encountered errors in the previous run.")

    else:
        force_reprocess = True  # No files → always process

    if force_reprocess and step_button("Start Transcription", key="transcribe_btn", allowed_states=["model_loaded","translation_completed","transcription_completed"], set_to="transcribing"):
        st.success("✅ Transcription step triggered.")

        whisper_task = "translate" if task == "Translate" else "transcribe"
        text, segments, info = transcribe_audio_file(
            st.session_state["whisper_model"],
            selected_file,
            task=whisper_task,
            beam_size=5,
            language=source_lang  
        )
        st.session_state["segments"] = segments
        st.session_state["text"] = text
        st.text_area("Full Transcript", text, height=200)
        st.success("✅ Transcription completed.")

        save_transcript(segments, output_base, formats=["txt", "srt", "ass", "vtt"])
        st.success("💾 Transcript saved in .txt, .srt, .vtt and .ass formats.")
        st.session_state.current_step = "transcription_completed"        






# --- Step 5: Translate Subtitles (Multi-file Support with Select All) ---
st.header("🌐 Step 5: Translate Subtitles")

if "processed_df" in st.session_state:
    subtitle_dir = Path(output_subtitle_dir)
    available_files = []

    for path in st.session_state["processed_df"]["processed_path"]:
        stem = Path(path).stem
        for ext in [".srt", ".ass", ".vtt"]:
            candidate = subtitle_dir / f"{stem}{ext}"
            if candidate.exists():
                available_files.append(candidate)

    if not available_files:
        st.warning("⚠️ No subtitle files (.srt/.ass/.vtt) found.")
    else:
        all_str = "🔘 Select All"
        display_options = [str(p) for p in available_files]
        selection = st.multiselect("Select subtitle files for translation", options=[all_str] + display_options)

        # Handle Select All logic
        files_to_translate = [Path(s) for s in display_options] if all_str in selection else [Path(s) for s in selection]

        if step_button("Translate Selected Subtitles", key="translate_multi_btn", allowed_states=["preprocessed","translation_completed","transcription_completed","batch_transcription_completed"], set_to="translating"):
            st.success("✅ Translation step triggered.")
            translator = Translator()
            translation_results = []

            for file_path in files_to_translate:
                suffix = file_path.suffix.lower()
                out_path = file_path.with_name(f"{file_path.stem}_{tgt_lang}{file_path.suffix}")

                if out_path.exists():
                    translation_results.append({"file": file_path.name, "status": "Skipped (already translated)"})
                    continue

                with st.spinner(f"Translating {file_path.name} to {tgt_lang}..."):
                    if suffix == ".srt":
                        translate_srt_file(file_path, translator, src_lang=src_lang, tgt_lang=tgt_lang, save_path=out_path)
                    elif suffix == ".ass":
                        translate_ass_file(file_path, translator, src_lang=src_lang, tgt_lang=tgt_lang, save_path=out_path)
                    elif suffix == ".vtt":
                        translate_vtt_file(file_path, translator, src_lang=src_lang, tgt_lang=tgt_lang, save_path=out_path)
                    else:
                        translation_results.append({"file": file_path.name, "status": "⚠️ Unsupported format"})
                        continue

                translation_results.append({"file": file_path.name, "status": "✅ Translated"})

            st.success("📤 Translation completed.")
            st.dataframe(pd.DataFrame(translation_results))
            st.session_state.current_step="translation_completed"
            



# --- Step 6: Batch Transcription (Whisper only) ---
st.header("📦 Step 6: Batch Processing (Transcription & Whisper Translation Only)")

batch_mode = st.checkbox("Enable Batch Transcription")
selected_formats = st.multiselect("Select Output Formats", ["txt", "srt", "ass", "vtt"], default=["srt", "txt"])
max_files = st.slider("Limit Number of Files (for testing)", 1, 50, 5)

if batch_mode and "processed_df" in st.session_state and "whisper_model" in st.session_state:
    to_process = st.session_state["processed_df"].head(max_files)

    force_batch = st.checkbox("🔁 Reprocess Already Done Files?")
    if force_batch:
        st.warning("⚠️ Only enable this if you changed the model size, selected the wrong language, or want to regenerate the subtitle formats.")

    if step_button("Run Batch Transcription", key="batch_transcribe_btn", allowed_states=["model_loaded","transcription_completed","translation_completed"], set_to="batch_processing"):
        st.success("✅ Batch transcription triggered.")
        progress_bar = st.progress(0)
        results = []

        for i, row in enumerate(to_process.itertuples()):
            input_path = Path(row.processed_path)
            whisper_task = "translate" if task == "Translate" else "transcribe"
            output_base = Path(output_subtitle_dir) / input_path.stem

            # Skip if all output formats already exist
            all_outputs_exist = all((output_base.with_suffix(f".{fmt}").exists() for fmt in selected_formats))

            if all_outputs_exist and not force_batch:
                results.append({
                    "file": input_path.name,
                    "language": "⏩ skipped",
                    "duration": "-",
                    "status": "Already Exists"
                })
                progress_bar.progress((i + 1) / len(to_process))
                continue

            text, segments, info = transcribe_audio_file(
                st.session_state["whisper_model"],
                input_path,
                task=whisper_task,
                language=source_lang  # pass user-selected language from sidebar
            )
            save_transcript(segments, output_base, formats=selected_formats)

            results.append({
                "file": input_path.name,
                "language": info.language,
                "duration": round(info.duration, 2),
                "status": "Done"
            })
            progress_bar.progress((i + 1) / len(to_process))

        st.success("✅ Batch processing complete!")
        st.dataframe(pd.DataFrame(results))
        st.session_state.current_step = "batch_transcription_completed"



if st.sidebar.button("🔄 Reset App"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# --- Footer ---
st.markdown("---")
st.caption("Made with ❤️ using OpenAI Whisper + Meta NLLB + Streamlit")
