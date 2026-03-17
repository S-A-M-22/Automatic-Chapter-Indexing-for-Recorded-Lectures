# #run terminal command to setup virtual environment first: python -3.11 -m venv .venv
# PYTHON_VERSION := 3.11
# VENV := .venv
# PIP := $(VENV)/bin/pip
# PY := $(VENV)/bin/python

# .PHONY: all setup install-system venv install-python run-notebook clean

# all: setup

# setup: install-system venv install-python

# venv:
# 	@echo "Creating virtual environment in $(VENV) (if missing)..."
# 	@test -d $(VENV) || python$(PYTHON_VERSION) -m venv $(VENV)
# 	$(PIP) install -U pip setuptools wheel

# install-system:
# 	sudo apt-get update -y
# 	sudo apt-get install -y \
# 		python$(PYTHON_VERSION) \
# 		python$(PYTHON_VERSION)-venv \
# 		python$(PYTHON_VERSION)-dev \
# 		ffmpeg \
# 		libgl1

# install-python:
# 	$(PIP) install -U \
# 		openai-whisper \
# 		torch torchaudio \
# 		numpy soundfile \
# 		transformers accelerate datasets[audio] \
# 		ffmpeg-python \
# 		opencv-python \
# 		jupyter nbconvert

# run-notebook:
# 	$(PY) -m nbconvert \
# 		--to notebook \
# 		--execute "Notebook 1.ipynb" \
# 		--output "Notebook-executed.ipynb" \
# 		--ExecutePreprocessor.timeout=1200

# clean:
# 	rm -rf $(VENV)
# 	rm -rf __pycache__
# 	rm -rf Notebook-executed.ipynb
# 	@echo "venv cleaned!\n"

PYTHON := python
VENV := .venv
PIP := $(VENV)/Scripts/pip
PY := $(VENV)/Scripts/python

.PHONY: all setup venv install-python run-notebook clean check-ffmpeg

all: setup

setup: check-ffmpeg venv install-python

venv:
	@if not exist $(VENV) ( \
		echo Creating virtual environment... && \
		$(PYTHON) -m venv $(VENV) \
	)
	$(PIP) install -U pip setuptools wheel

check-ffmpeg:
	@ffmpeg -version >nul 2>&1 || ( \
		echo FFmpeg NOT found. Install FFmpeg and add to PATH. && exit 1 \
	)

install-python:
	$(PIP) install -U \
		openai-whisper \
		torch torchaudio \
		numpy soundfile \
		transformers accelerate datasets[audio] \
		ffmpeg-python \
		opencv-python \
		jupyter nbconvert

run-notebook:
	$(PY) -m nbconvert \
		--to notebook \
		--execute "Notebook 1.ipynb" \
		--output "Notebook-executed.ipynb" \
		--ExecutePreprocessor.timeout=1200

clean:
	@if exist $(VENV) rmdir /s /q $(VENV)
	@if exist __pycache__ rmdir /s /q __pycache__
	@if exist Notebook-executed.ipynb del Notebook-executed.ipynb
	@echo venv cleaned!
