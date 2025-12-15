# Evaluation Checklist

Use this checklist to ensure all evaluation components are ready before running.

## Pre-Flight Checks

### 1. Test Queries File ✓
- **File**: `scripts/test_queries.json`
- **Status**: Valid JSON with 8 queries
- **Structure**: 
  - 3 queries with `reference_answer` (for ROUGE/EM)
  - 5 queries without `reference_answer` (retrieval metrics only)
- **Validation**: Run `python3 -c "import json; json.load(open('scripts/test_queries.json'))"`

### 2. Dependencies ✓
Required packages:
- `rouge-score` - for ROUGE metrics
- `matplotlib` - for visualizations
- `numpy` - for numerical operations

Install with:
```bash
pip install rouge-score matplotlib numpy
```

### 3. File Paths ✓
Ensure these files exist:
- `scripts/test_queries.json`
- `scripts/evaluate_rag_system.py`
- `scripts/create_visualizations.py`
- `scripts/fill_results_tables.py`
- `overleaf-report/AI_PROJECT_REPORT/experiements/results.tex`

### 4. Data and Indices ✓
Ensure:
- FAISS indices are built: `data/indices/` contains index files
- Course data is processed: `data/processed/` contains course materials
- LLM is configured: `.env` has `GEMINI_API_KEY` or Ollama is running

## Evaluation Workflow

### Step 1: Run Evaluation
```bash
python3 scripts/evaluate_rag_system.py \
  --queries scripts/test_queries.json \
  --output scripts/evaluation_results.json
```

**Expected Output**:
- Console output with metrics for each query
- Summary tables showing:
  - Answer Generation Quality (ROUGE scores)
  - Semantic Similarity Metrics
  - Retrieval Performance Metrics
  - Query Rewriting Improvement
  - System Performance
- JSON file: `scripts/evaluation_results.json`

**What to Check**:
- ✓ All queries processed successfully
- ✓ Metrics calculated (some may show "N/A" if no reference answers)
- ✓ No errors in console output

### Step 2: Generate Visualizations
```bash
python3 scripts/create_visualizations.py \
  --results scripts/evaluation_results.json \
  --output-dir overleaf-report/AI_PROJECT_REPORT/figures
```

**Expected Output**:
- `figures/semantic_similarity.pdf`
- `figures/precision_at_k.pdf`
- `figures/query_rewriting_improvement.pdf`

**What to Check**:
- ✓ All 3 PDF files created
- ✓ Files are not empty
- ✓ Can open PDFs to verify charts

### Step 3: Fill LaTeX Tables
```bash
python3 scripts/fill_results_tables.py \
  --results scripts/evaluation_results.json \
  --output overleaf-report/AI_PROJECT_REPORT/experiements/results.tex
```

**Expected Output**:
- Updated `results.tex` with filled tables:
  - `tab:answer_quality`
  - `tab:retrieval_metrics`
  - `tab:query_rewriter_impact`
  - `tab:performance`

**What to Check**:
- ✓ Tables updated (no "To be filled" values)
- ✓ Numbers match evaluation output
- ✓ LaTeX compiles without errors

## Troubleshooting

### Issue: "rouge-score not installed"
**Solution**: `pip install rouge-score`

### Issue: "matplotlib not installed"
**Solution**: `pip install matplotlib numpy`

### Issue: "No module named 'config'"
**Solution**: Run from project root directory

### Issue: "FAISS indices not found"
**Solution**: Run `python run.py --index` to build indices first

### Issue: "LLM client initialization failed"
**Solution**: 
- Check `.env` file has `GEMINI_API_KEY` (for Gemini)
- Or ensure Ollama is running (for Ollama)
- Or evaluation will use retrieval-only mode

### Issue: "Table not found in results.tex"
**Solution**: Ensure `results.tex` has the correct table labels:
- `tab:answer_quality`
- `tab:retrieval_metrics`
- `tab:query_rewriter_impact`
- `tab:performance`

### Issue: Metrics show "N/A"
**Expected**: This is normal for queries without `reference_answer`. The script will:
- Skip ROUGE/EM for those queries
- Still calculate retrieval metrics and semantic similarity

## Verification

After running all steps, verify:

1. **Evaluation Results**:
   - `scripts/evaluation_results.json` exists and is valid JSON
   - Contains `metrics` and `detailed_results` sections

2. **Visualizations**:
   - 3 PDF files in `overleaf-report/AI_PROJECT_REPORT/figures/`
   - Files open correctly and show charts

3. **LaTeX Tables**:
   - All tables in `results.tex` are filled
   - Numbers match evaluation output
   - LaTeX compiles successfully

4. **Figures in LaTeX**:
   - Figures reference correct paths
   - Captions are appropriate
   - Figures appear in correct sections

## Quick Test

Run this to test the setup:
```bash
# Validate setup
python3 scripts/validate_evaluation_setup.py

# If validation passes, run evaluation
python3 scripts/evaluate_rag_system.py --queries scripts/test_queries.json
```

## Notes

- Evaluation may take 10-30 minutes depending on number of queries and LLM response time
- Some metrics may be "N/A" if reference answers are unavailable (this is expected)
- Visualization script requires matplotlib (non-interactive backend)
- All scripts handle missing metrics gracefully

