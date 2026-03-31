DATA PREPARATION CHECKLIST
===========================

PHASE 1: PLANNING & COLLECTION
─────────────────────────────
□ Decide data source strategy (PhishTank, Kaggle, internal emails, etc.)
□ Aim for 5,000-10,000 new examples
□ Target: 50% phishing, 50% legitimate (balance matters!)
□ Plan 1-2 weeks for collection if doing this manually

PHASE 2: FORMATTING
──────────────────
□ Create CSV file with three columns: subject, body, label
  OR two columns: body, label (subject will be auto-generated from body)
□ Label: 0 = legitimate, 1 = phishing
□ Remove sender/recipient names (anonymize PII)
□ Save as: datasets/training_data/new_training_data.csv
□ Use UTF-8 encoding

EXAMPLE ROW:
"Meeting Reminder","Our team sync is scheduled for tomorrow at 2pm in Conference Room 4",0
"URGENT CONFIRM","Your account has been locked. Click here immediately to verify your identity",1

PHASE 3: CLEANING
────────────────
□ Remove email headers (From:, To:, Date:, Subject: prefixes)
□ Clean up extra whitespace and formatting
□ Check minimum length: 10 characters
□ Check maximum length: 10,000 characters
□ Ensure all required fields are present

PHASE 4: VALIDATION
───────────────────
□ Run validation script on the new data
□ Check for parse errors (0 errors expected)
□ Verify label distribution (ratio should be 1.0-1.5)
□ Verify minimum 1,000 examples per class
□ Check: total > 1,000 examples (5,000+ is better)

VALIDATION COMMAND:
  python prepare_data.py --validate new_training_data.csv

PHASE 5: MERGING (OPTIONAL)
───────────────────────────
□ Decide: Use NEW data only, or MERGE with existing?
  - Option A: Use new data only (if 5,000+ examples and high quality)
  - Option B: Merge new + existing for more diversity
□ If merging, run merge command
□ Validate merged dataset

MERGE COMMAND:
  python prepare_data.py --merge \
    datasets/phishing_email_datasets/enron_phishing_and_legitimate_email_dataset.csv \
    datasets/training_data/new_training_data.csv \
    datasets/training_data/combined_training_dataset.csv

PHASE 6: FINAL CHECKS
─────────────────────
□ Run final validation on merged/final dataset
□ Confirm balance: 40-60% phishing is acceptable
□ Confirm no errors reported
□ Dataset is ready for training

FINAL VALIDATION COMMAND:
  python prepare_data.py --validate datasets/training_data/combined_training_dataset.csv

PHASE 7: PREPARE FOR TRAINING
──────────────────────────────
□ Copy final dataset to correct location
□ Update train_models.py to use new dataset path
□ Run training script
□ Monitor progress with monitor_training_progress.py

─────────────────────────────────────────────────────────────────

QUICK REFERENCE: DATA REQUIREMENTS
───────────────────────────────────

Minimum:
  • 1,000 total examples (500 phishing + 500 legitimate)

Recommended:
  • 5,000-10,000 total examples
  • 50% phishing, 50% legitimate
  • Recent samples (2023-2026)
  • Diverse phishing types (urgency, financial, credential, delivery)

Optimal:
  • 10,000+ examples
  • 45-55% class balance
  • Mix of old (2023) and new (2024-2026) samples
  • Include domain-specific emails (your organization's patterns)

─────────────────────────────────────────────────────────────────

DATA QUALITY CRITERIA
─────────────────────

REJECT if:
  ✗ Label is not 0 or 1
  ✗ Body is empty or missing
  ✗ Text is < 10 characters
  ✗ Text is > 10,000 characters
  ✗ Parsing errors detected
  ✗ Class imbalance ratio > 2.0

ACCEPT if:
  ✓ All labels are 0 or 1
  ✓ All bodies are present and valid
  ✓ Text length: 10-10,000 characters
  ✓ No parsing errors
  ✓ Class balance ratio 1.0-1.5
  ✓ Minimum 1,000 examples per class

─────────────────────────────────────────────────────────────────

RECOMMENDED DATA SOURCES
────────────────────────

PHISHING:
  1. PhishTank (phishtank.com)
     - Free, regularly updated
     - Downloadable CSV/JSON
     - ~53,000+ phishing entries available
     
  2. Kaggle Datasets
     - Search: "phishing emails" or "phishing dataset"
     - Many community-created datasets
     - Download limit: Check individual dataset terms
     
  3. Your Organization
     - Emails marked as phishing by users or security team
     - Real patterns specific to your domain
     - MUST anonymize before use
     
  4. APWG (Anti-Phishing Working Group)
     - Public phishing reports
     - Industry-wide samples
     
  5. GitHub Phishing Repos
     - Search: "phishing-emails" or "phishing-dataset"
     - Community-curated datasets

LEGITIMATE:
  1. Your Own Inbox
     - Real business emails
     - Authentic tone and patterns
     - Anonymize sender/recipient
     
  2. Company Email Archives
     - Team communications
     - HR/IT announcements
     - Project discussions
     
  3. Public Corpora
     - Enron (already have)
     - Lingspam
     - Jester dataset
     
  4. Generated Samples
     - Common business email templates
     - Meeting confirmations, project updates
     - Policy announcements, newsletters

─────────────────────────────────────────────────────────────────

CSV FORMAT TEMPLATE
───────────────────

subject,body,label
"Subject 1","Body text can span multiple lines. Use quotes if text contains commas, quotes, or newlines.",0
"URGENT: ACTION REQUIRED","Your account will be suspended. Click here now: https://phishing-site.com",1
"Team Meeting","Quick reminder: standup is at 10am tomorrow.",0

Rules:
  • First row is header: subject,body,label
  • Each row is one email
  • Columns separated by commas
  • Text values quoted with double quotes
  • If text contains quotes, escape them: "He said ""hello"""
  • In Excel: Save As → CSV (Comma-delimited)
  • In Python: Use csv.DictWriter with proper encoding

─────────────────────────────────────────────────────────────────

ANONYMIZATION CHECKLIST
───────────────────────
Before including emails in training data:

□ Remove sender name/email: "From: John Smith <john@company.com>" → Remove
□ Remove recipient names/emails: "To: team@company.com" → Remove/Generalize
□ Mask company names if sensitive: "Acme Corp" → "[COMPANY]"
□ Mask email addresses: "john@acme.com" → "[EMAIL]"
□ Mask IP addresses: "192.168.1.1" → "[IP]"
□ Mask phone numbers: "+1-555-123-4567" → "[PHONE]"
□ Keep URLs as-is (important for phishing detection)
□ Keep technical details (important for legitimacy assessment)

Example:
BEFORE: "Hi John, our meeting at 1pm with James from acme-payments is confirmed. Call me at 555-1234."
AFTER:  "Hi [NAME], our meeting at 1pm with [NAME] from [COMPANY] is confirmed. Call me at [PHONE]."

─────────────────────────────────────────────────────────────────

TROUBLESHOOTING
───────────────

Problem: "CSV parse error"
Solution: Check for unescaped quotes or newlines. All text should be quoted.

Problem: "Class imbalance too high"
Solution: Collect more of the underrepresented class. Aim for 45-55% split.

Problem: "Text length validation failed"
Solution: Remove entries < 10 chars (likely garbage) or > 10,000 chars (likely forwarded threads).

Problem: "No errors but accuracy still low"
Solution: May need more data quality curation or domain-specific samples. Consider:
  - Reviewing misclassified examples
  - Adding more recent phishing variants
  - Collecting organization-specific emails

─────────────────────────────────────────────────────────────────
