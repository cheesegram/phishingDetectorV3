DATA PREPARATION SUMMARY & QUICK START
======================================

You have 3 files to help you prepare data:
1. prepare_data.py       - Functions to clean, validate, and merge data
2. DATA_PREPARATION_CHECKLIST.md - Detailed checklist and requirements
3. This file             - Quick start and summary

═══════════════════════════════════════════════════════════════════════════════

QUICK START (5 STEPS)
════════════════════

Step 1: COLLECT DATA (1-2 weeks)
────────────────────────────────
   Goal: Get 5,000-10,000 new email examples
   
   Option A - Easy (Use existing public datasets):
   • PhishTank: www.phishtank.com (download CSV)
   • Kaggle: Search "phishing emails" → Download dataset
   • GitHub: Search phishing-datasets → Clone/download
   Time: 1 day to download and prepare
   
   Option B - Best (Use your organization's emails):
   • Export 2,000-5,000 emails from your company mailbox
   • Mix legitimate (HR, team, project emails) 
      + phishing (marked by users as junk/phishing)
   • Anonymize names and addresses
   Time: 3-5 days to collect and clean

Step 2: FORMAT DATA (1-2 hours)
───────────────────────────────
  Create a CSV file with either format:
   
  Option A (recommended):
   subject,body,label
   "Meeting reminder","Team sync at 2pm tomorrow",0
   "URGENT: Verify Account","Click here now: https://phishing.com",1

  Option B (supported):
  body,label
  "Meeting reminder Team sync at 2pm tomorrow",0
  "URGENT: Verify Account Click here now: https://phishing.com",1

  If subject is missing, prepare_data.py will auto-generate subject from body.
   
   Save as: datasets/training_data/new_training_data.csv
   
   CSV formatting tips:
   • Use UTF-8 encoding (most editors default)
   • Excel: File → Save As → Format: CSV (Comma delimited)
   • Google Sheets: File → Download → CSV
   • Python: Use csv.DictWriter module

Step 3: VALIDATE DATA (30 minutes)
──────────────────────────────────
   Run this command to check for errors:
   
   $ python prepare_data.py
   
   This will:
   ✓ Check CSV is properly formatted
   ✓ Verify all labels are 0 or 1
   ✓ Check text length is valid (10-10,000 chars)
   ✓ Report class balance (should be ~50/50)
   ✓ Show any errors found
   
   Call from Python:
   >>> from prepare_data import analyze_csv_dataset
   >>> analyze_csv_dataset('datasets/training_data/new_training_data.csv')
   
   Expected output (good):
   ✓ Total Records: 5,000
   ✓ Legitimate: 2,500 (50.0%)
   ✓ Phishing: 2,500 (50.0%)
   ✓ Data is well balanced
   ✓ No validation errors found

Step 4: MERGE WITH EXISTING (Optional, 30 min)
───────────────────────────────────────────────
   If you want to combine new data with existing Enron dataset:
   
   >>> from prepare_data import merge_csv_files
   >>> merge_csv_files(
   ...     'datasets/phishing_email_datasets/enron_phishing_and_legitimate_email_dataset.csv',
   ...     'datasets/training_data/new_training_data.csv',
   ...     'datasets/training_data/combined_training_dataset.csv'
   ... )
   
   This creates: combined_training_dataset.csv
   
   Or use just the new data if it's high quality (5,000+ examples).

Step 5: TRAIN MODEL (1-2 hours)
───────────────────────────────────
   TODO: We'll create an improved training script
   that accepts the prepared CSV file.
   
   For now, you have the data ready to train.

═══════════════════════════════════════════════════════════════════════════════

MINIMUM REQUIREMENTS
════════════════════

For retraining to improve model accuracy, you need:

Data Size:
  • Minimum: 1,000 examples (500 phishing, 500 legitimate)
  • Recommended: 5,000-10,000 examples
  • Optimal: 10,000+ examples

Class Balance (CRITICAL):
  • Minimum acceptable: 40-60% each class
  • Recommended: 45-55% each class
  • Ideal: 50-50 split
  • DO NOT use 90% phishing + 10% legitimate (won't train well)

Data Quality:
  • Each email: 10-10,000 characters
  • Valid labels: 0 or 1 only
  • No parsing errors
  • No PII (anonymized names, emails, phone numbers)
  • Recent samples preferred (2023-2026)

═══════════════════════════════════════════════════════════════════════════════

CSV FORMAT REFERENCE
════════════════════

CORRECT:
────────
subject,body,label
"Q1 Planning","Let's meet tomorrow to discuss Q1 deliverables.",0
"URGENT ACTION REQUIRED","Your account has been locked. Verify now: https://secure-verify.com/login",1
"Team Standup","Reminder: standup is at 10am in Conference Room B.",0

INCORRECT (will fail validation):
──────────────────────────────────
subject,body,label
"Meeting","",0  ← FAIL: body is empty
"Alert","Your account","abc"  ← FAIL: label is not 0 or 1
"Email","x",0  ← FAIL: text too short (minimum 10 chars)
"Subject"with quote,"Body has unescaped quote",0  ← FAIL: CSV parsing error

FIXES:
──────
1. Empty body → Remove row or add text
2. Invalid label → Must be 0 (legitimate) or 1 (phishing)
3. Too short → Remove rows or combine with other text
4. CSV parsing → Quote all text, escape quotes: "He said ""hello"""

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE WORKFLOW
════════════════

From download to validation:

# 1. Download PhishTank dataset
curl https://data.phishtank.com/data/online-valid.csv.gz > phishtank.csv.gz
gunzip phishtank.csv.gz

# 2. Extract emails from your inbox (Gmail example)
# Use Google Takeout or export from Outlook
# Extract subject and body columns

# 3. Combine into single CSV with format:
# subject,body,label

# 4. Validate
python
>>> from prepare_data import analyze_csv_dataset
>>> analyze_csv_dataset('my_training_data.csv')

# 5. If validation passes, you're ready to train
# If validation fails, review errors and fix

═══════════════════════════════════════════════════════════════════════════════

TROUBLESHOOTING
═══════════════

Issue: "No such file or directory"
→ Check file path is correct
→ Use absolute path: /full/path/to/file.csv

Issue: "CSV parse error"
→ Check for unescaped quotes in text
→ All text should be quoted: "subject","body",0
→ Quotes in text should be doubled: "He said ""hello"""

Issue: "Invalid label"
→ Labels must be 0 or 1
→ Check no extra spaces: "0 " or " 1" will fail
→ Use formula to convert: =IF(PHISHING, 1, 0)

Issue: "Class imbalance too high"
→ You have too much phishing or too much legitimate
→ Collect more of the minority class
→ Or randomly remove samples from majority class

Issue: "Text too long/too short"
→ Minimum: 10 characters
→ Maximum: 10,000 characters
→ Remove outliers or combine short emails

═══════════════════════════════════════════════════════════════════════════════

FILE LOCATIONS
══════════════

Place prepared data here:
  datasets/training_data/new_training_data.csv

Existing data (reference):
  datasets/phishing_email_datasets/enron_phishing_and_legitimate_email_dataset.csv

Combined data output:
  datasets/training_data/combined_training_dataset.csv

Scripts:
  prepare_data.py - Data validation and merging
  test_model.py - Model evaluation
  train_models.py - Model training (to be updated)

═══════════════════════════════════════════════════════════════════════════════

NEXT ACTIONS
════════════

1. Read the DATA_PREPARATION_CHECKLIST.md for detailed requirements
2. Decide data source (PhishTank, Kaggle, your inbox, or combination)
3. Collect 5,000-10,000 emails
4. Format as CSV: subject, body, label
5. Save to: datasets/training_data/new_training_data.csv
6. Run validation: python prepare_data.py
7. Once validation passes, let me know and I'll set up training script

═══════════════════════════════════════════════════════════════════════════════
