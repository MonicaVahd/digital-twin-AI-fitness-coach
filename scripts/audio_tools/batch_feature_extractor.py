import os
import subprocess

# === UPDATE THESE PATHS IF NEEDED ===
SMILEXTRACT_PATH = '/mnt/d/projects/digital-twin-ai-fitness-coach/opensmile-master/opensmile-master/build/progsrc/smilextract/SMILExtract'
CONFIG_PATH = '/mnt/d/projects/digital-twin-ai-fitness-coach/opensmile-master/opensmile-master/config/gemaps/v01a/GeMAPSv01a.conf'
AUDIO_ROOT = 'user_emotion_label'         # Relative to your current working directory
OUTPUT_ROOT = 'opensmile_features'        # Relative to your current working directory
# =====================================

for root, dirs, files in os.walk(AUDIO_ROOT):
    for file in files:
        if file.lower().endswith('.wav'):
            input_path = os.path.join(root, file)
            # Build output path, preserving subfolder structure
            rel_path = os.path.relpath(root, AUDIO_ROOT)
            output_dir = os.path.join(OUTPUT_ROOT, rel_path)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file.replace('.wav', '.csv'))
            cmd = [
                SMILEXTRACT_PATH,
                '-C', CONFIG_PATH,
                '-I', input_path,
                '-O', output_path
            ]
            print(f"Processing {input_path} -> {output_path}")
            try:
                subprocess.run(cmd, check=True)
            except Exception as e:
                print(f"Error processing {input_path}: {e}")