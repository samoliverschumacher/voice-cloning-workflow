.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

.PHONY: show_current_dir
show_current_dir:
	@echo $(CURDIR)

# Use bash syntax
SHELL := /bin/bash
PYTHON_INTERPRETER = python3

# Specify directory structure of a single data (multi-speaker audio file like a podcast)
TRANSCRIPT-ID = season-1/episode-12# Define the URL extension id to scrape the transcript. (i.e. www.bbc.com/podcasts/trainscript-id)
DATA-IDENTIFIER = 2018_s1e12# Unique identifier for the data: <Group>_<Name>
DATA_GROUP_NAME = $(shell echo $(DATA-IDENTIFIER) | cut -d_ -f1)
DATA_NAME = $(shell echo $(DATA-IDENTIFIER) | cut -d_ -f2)

RAW_DATA_DIR = $(CURDIR)/data/00_raw
INTERIM_DATA_DIR = $(CURDIR)/data/01_interim

RAW_AUDIO_FILE = $(RAW_DATA_DIR)/podcasts/$(DATA_GROUP_NAME)/$(DATA_NAME).wav

DATA_FOLDER = $(INTERIM_DATA_DIR)/$(DATA-IDENTIFIER)
AUDIO_FILE = $(INTERIM_DATA_DIR)/$(DATA-IDENTIFIER)/audio.wav
DIARY_FILE = $(INTERIM_DATA_DIR)/$(DATA-IDENTIFIER)/diary.csv
METADATA_FILE = $(INTERIM_DATA_DIR)/$(DATA-IDENTIFIER)/metadata.json

# Define the variable for the processed transcript
TRANSCRIPT_FILE = $(INTERIM_DATA_DIR)/$(DATA-IDENTIFIER)/transcript.txt

define format_var # Pretty print the variable name and its value
	@printf "\e[1m\e[32m$(1)\e[0m\e[3m '$(2)' \e[0m\n";
endef

printvars:
	$(call format_var, Transcript ID: ,$(TRANSCRIPT-ID))
	$(call format_var, Data Identifier: ,$(DATA-IDENTIFIER))
	$(call format_var, Group Name: ,$(DATA_GROUP_NAME))
	$(call format_var, Filename: ,$(DATA_NAME))
	$(call format_var, Raw Data Folder: ,$(RAW_DATA_DIR))
	$(call format_var, Interim Data Folder: ,$(INTERIM_DATA_DIR))
	$(call format_var, Raw Audio File: ,$(RAW_AUDIO_FILE))
	$(call format_var, Audio & Transcripts Folder: ,$(DATA_FOLDER))
	$(call format_var, Audio File: ,$(AUDIO_FILE))
	$(call format_var, Diary File: ,$(DIARY_FILE))
	$(call format_var, Metadata File: ,$(METADATA_FILE))
	$(call format_var, Transcript File: ,$(TRANSCRIPT_FILE))

all: setup download diarize check-diary split-transcript split-audio wpm # setup -> download -> diarize -> check-diary -> split-transcript -> split-audio -> wpm

setup: # Initialises new directory, moves raw audiofile into it
	$(PYTHON_INTERPRETER) scripts/utils.py ini-file-dir $(DATA-IDENTIFIER) $(RAW_AUDIO_FILE)

download: $(DATA_FOLDER) # Gather transcript, download
	$(PYTHON_INTERPRETER) scripts/crawl.py $< $(TRANSCRIPT-ID) $(DATA_GROUP_NAME)

diarize: $(RAW_AUDIO_FILE) $(TRANSCRIPT_FILE) # Use pyannote-audio to create diary file from speaker turns predicted in audio file.
	$(PYTHON_INTERPRETER) scripts/diarize.py "$$($(PYTHON_INTERPRETER) scripts/utils.py extract-names $(TRANSCRIPT_FILE))" $< --output_file $(DIARY_FILE) -cr --check $(TRANSCRIPT_FILE)

check-diary: $(DIARY_FILE) # Perform checks: diary speakers equal to raw transcript names
	$(PYTHON_INTERPRETER) scripts/transcript_analysis.py check-speaker-order $< --transcript $(TRANSCRIPT_FILE)

split-transcript: $(TRANSCRIPT_FILE) # Processes text so it contains only valid tokens, the splits into separate files for each speaker turn
	$(PYTHON_INTERPRETER) scripts/process_transcript.py $<

wpm: # Perform checks: Words per minute (wpm) spoken in each speaker-turn, according to the diarization
	@echo "High words per minute (>250) with greater than 20 words;"
	$(PYTHON_INTERPRETER) scripts/transcript_analysis.py wpm $(DIARY_FILE) -wpm ">250" -wc ">20"  | column -t -s,
	@echo ""
	@echo "Files with less than 20 words;"
	$(PYTHON_INTERPRETER) scripts/transcript_analysis.py wpm $(DIARY_FILE) -wc "<20"  | column -t -s,

add-metadata: # Command line interface to append text to metadata file
	@echo "Adding METADATA to $(METADATA_FILE)"
	$(PYTHON_INTERPRETER) scripts/utils.py append-metadata $(METADATA_FILE)

split-audio: $(AUDIO_FILE) # Splits the audio file into speaker turns based on diarization
	$(PYTHON_INTERPRETER) scripts/split_wav.py split-audio $< --from_diary "$(DIARY_FILE)"

align: $(DATA_FOLDER) # Perform forced alignments on all speech-transcript pairs, to get word-alignments
	$(PYTHON_INTERPRETER) scripts/forced_alignment.py align $<


%-alignments.json: %.wav %.processed.txt # Perform forced alignments on a single pair of speech-transcript files, to get word-alignments
	$(PYTHON_INTERPRETER) scripts/forced_alignment.py align-single $^

align-single: $(SPEECH_FILES) $(WORD_FILES)
	$(PYTHON_INTERPRETER) scripts/forced_alignment.py align $<

spm: # Check the alignments for slowly spoken syllables
	$(PYTHON_INTERPRETER) scripts/transcript_analysis.py spm $(DATA_FOLDER)

word-score: # Print word alignment score summaries of all alignment files
	$(PYTHON_INTERPRETER) scripts/transcript_analysis.py calc-wordscores $(DATA_FOLDER) | column -t -s,

# TODO: write more atomic version of this, so that make can take care of skipping outdated utterance files, instead of it happening in python
split-speaker-turns: # Split into uttterances shorter than the speaker turn.
	$(PYTHON_INTERPRETER) scripts/split_wav.py split-waveform-cli $(DATA_FOLDER) $(DATA_FOLDER)


###########################################################################
#				COMMANDS FOR ALL DATASETS   							  #
###########################################################################

RTVC_REPO_DIR = $(subst $(notdir $(CURDIR)),,$(CURDIR))Real-Time-Voice-Cloning
FINAL_DESTINATION := $(RTVC_REPO_DIR)/datasets/Podcasts
CREATED_DIRECTORIES := $(shell find $(FINAL_DESTINATION) -type f -name 'transcript.alignment.txt' -exec dirname {} \; | sort -u)

generate-alignments:  # Create index mapping processed content to timestamps, for each file.
	$(PYTHON_INTERPRETER) scripts/move_files.py generate-alignments

move-datasets: generate_alignments  # Move processed datasets to model training folder
	$(PYTHON_INTERPRETER) scripts/move_files.py move

.PHONY: check_move_ok
check_move_ok: move_datasets  # Run checks on moved training datasets to check validity
	@$(foreach dir,$(CREATED_DIRECTORIES),./scripts/checks.sh $(dir);)

clean:
	rm -rf __pycache__
	rm -rf scripts/__pycache__
