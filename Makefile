# ===========================================================
#  AutoChapter  -  Makefile (Windows / Python 3.11)
#
#  Quick start:
#      make            # create venv + install packages
#      make run        # launch Flask app (text-first pipeline)
#      make run-visual # launch Flask app (visual sliding-window pipeline)
#      make clean      # remove venv and caches
# ===========================================================

PYTHON_311 := C:/Python311/python.exe
VENV       := .venv
PY         := $(VENV)/Scripts/python.exe
PIP        := $(PY) -m pip
PORT       := 5000

.PHONY: all setup venv install check-python check-ffmpeg install-ffmpeg run run-visual clean

# -- Default target ------------------------------------------
all: setup

# -- Full setup (venv + packages) ----------------------------
setup: check-python check-ffmpeg venv install
	@echo.
	@echo =============================================
	@echo   Setup complete!  Run  make run  to start.
	@echo =============================================

# -- Pre-flight checks --------------------------------------
check-python:
	@$(PYTHON_311) --version >nul 2>&1 || ( \
		echo ERROR: Python 3.11 not found at $(PYTHON_311). && \
		echo Install Python 3.11 or update PYTHON_311 in this Makefile. && \
		exit /b 1 \
	)
	@echo [OK] Python 3.11 found

check-ffmpeg:
	@ffmpeg -version >nul 2>&1 && ( \
		echo [OK] FFmpeg found \
	) || ( \
		echo [!!] FFmpeg not found - installing via choco ... && \
		choco install ffmpeg -y && \
		refreshenv && \
		echo [OK] FFmpeg installed \
	)

# -- Create virtual-env (idempotent) ------------------------
venv:
	@if not exist $(PY) ( \
		echo Creating .venv with Python 3.11 ... && \
		$(PYTHON_311) -m venv $(VENV) \
	)
	$(PIP) install --upgrade pip setuptools wheel

# -- Install all dependencies --------------------------------
install:
	$(PIP) install -r requirements.txt

# -- Run the Flask app (text-first pipeline) -------------------
run:
	@echo =============================================
	@echo   AutoChapter  http://localhost:$(PORT)
	@echo =============================================
	$(PY) run_app.py

# -- Run the Flask app (visual sliding-window pipeline) --------
run-visual:
	@echo =============================================
	@echo   AutoChapter (Visual Pipeline)
	@echo   http://localhost:$(PORT)
	@echo =============================================
	$(PY) sw_visual_app.py

# -- Housekeeping --------------------------------------------
clean:
	@if exist $(VENV) rmdir /s /q $(VENV)
	@if exist __pycache__ rmdir /s /q __pycache__
	@echo Cleaned.
