# Makefile — Medical AI Reliability Auditor
# Run any target with:  make <target>

.PHONY: install data train eval demo test paper clean

# ── Setup ─────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Full pipeline (no API key, template responses) ────────────────
run:
	python experiments/run_pipeline.py

# ── Full pipeline with OpenAI ─────────────────────────────────────
run-openai:
	python experiments/run_pipeline.py --openai

# ── Train BioBERT ─────────────────────────────────────────────────
train-biobert:
	python experiments/run_pipeline.py --model biobert

# ── Data only ─────────────────────────────────────────────────────
data:
	python experiments/run_pipeline.py --skip_train

# ── Train only (assumes data exists) ─────────────────────────────
train:
	python experiments/run_pipeline.py --skip_data

# ── Evaluate only ─────────────────────────────────────────────────
eval:
	python evaluation/evaluate_all.py

# ── Rule-based eval only (no GPU needed) ─────────────────────────
eval-rules:
	python evaluation/evaluate_all.py --rules_only

# ── Interactive demo ──────────────────────────────────────────────
demo:
	python demo.py

demo-examples:
	python demo.py --examples

demo-model:
	python demo.py --checkpoint experiments/best_bert.pt --examples

# ── Run unit tests ────────────────────────────────────────────────
test:
	python -m pytest tests/test_all.py -v

test-fast:
	python -m pytest tests/test_all.py -v -k "not TestModel"

# ── Hyperparameter search ─────────────────────────────────────────
hparam:
	python experiments/hyperparameter_search.py --epochs 2

# ── Plot training curves ──────────────────────────────────────────
plot:
	python experiments/plot_training.py

# ── Data exploration notebook ─────────────────────────────────────
explore:
	cd notebooks && python exploration.py

# ── Build paper PDF (requires pdflatex) ──────────────────────────
paper:
	cd paper && pdflatex paper.tex && pdflatex paper.tex
	@echo "PDF built: paper/paper.pdf"

# ── Clean generated files ─────────────────────────────────────────
clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -f experiments/hparam_run*.pt
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.toc

clean-all: clean
	rm -f data/raw_dataset.csv data/labeled_dataset.csv
	rm -f data/train_split.csv data/val_split.csv data/test_split.csv
	rm -f experiments/*.pt experiments/*.json experiments/*.png
	rm -f evaluation/*.png evaluation/*.csv evaluation/*.json
