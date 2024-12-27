# Automatic Cot-Caught Merger Detection

This repository contains code for automatically detecting the cot-caught merger in American English dialects using a machine learning approach.

## Overview

This project uses a random forest classifier to identify speakers with the cot-caught merger, a phonological feature distinguishing some North American dialects. The model is trained using formant frequencies and vowel length from speech recordings. The cot-caught merger is a phonological shift where the vowels /ɑ/ (as in "cot") and /ɔ/ (as in "caught") are pronounced the same.

## Data

The data comes from the **DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus**, available at https://catalog.ldc.upenn.edu/LDC93S1. **Dialect region 6 is New York City (no merger) and dialect region 7 is the Western US (merger)**.

### Usage

1.  **Clone the repository.**
2.  **Install the required libraries** using `pip install -r requirements.txt`.
3.  **Prepare your data:**
    *   **TextGrid files** must be named in the following format: `dialectregion_gender+speakerID_sentence#.TextGrid`. For example: `DR7_MWRE0_SX337.TextGrid`.
        *   `DR6` indicates the non-merger dialect region, and `DR7` indicates the merger region.
    *   **Sound files** (.wav) must have the same name as the TextGrid files, except with the `.wav` extension, for example `DR6_m123_1.wav`
        *   The sound files should correspond to the TextGrid files in terms of the speech recording each contains.
4.  **Run the script** `process_and_train.py`:

    ```bash
    python process_and_train.py --textgrid_dir <path to textgrid dir> --sound_dir <path to sound dir>
    ```
    Replace `<path to textgrid dir>` and `<path to sound dir>` with the actual paths to your directories containing the TextGrid and sound files, respectively.