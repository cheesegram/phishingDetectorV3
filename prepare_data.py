"""
DATA PREPARATION GUIDE FOR BERT RETRAINING
==========================================

This guide walks you through collecting, formatting, and validating email data
for retraining the BERT phishing detector model.

Steps:
1. Collect/source new phishing and legitimate emails
2. Format and clean the data
3. Create a unified CSV dataset
4. Validate quality and balance
5. Prepare for training
"""

# ============================================================================
# STEP 1: DATA COLLECTION SOURCES
# ============================================================================

"""
PHISHING DATA SOURCES:
- PhishTank (phishtank.com) - free phishing database, downloadable
- VirusTotal URLhaus - malicious URLs and associated emails
- Your own organization - collect misclassified emails from your current system
- Kaggle datasets - search "phishing emails" or "phishing dataset"
- SpamAssassin corpus - publicly available spam/phishing samples
- GitHub phishing-datasets repositories

LEGITIMATE DATA SOURCES:
- Gmail/Outlook inbox - your real work emails (anonymized)
- Internal company communications - HR, IT, team emails
- Newsletter archives - marketing, news, updates
- Public email corpora - Enron (already have), Lingspam, etc.
- GitHub email datasets - legitimate mail samples

COLLECTION TIPS:
✓ Aim for 5,000-10,000 new examples total (50% phishing, 50% legitimate)
✓ Balance is critical - don't collect 9,000 phishing + 100 legitimate
✓ Include recent samples (2023-2026) to capture modern phishing patterns
✓ Include variety: urgency, financial, credential theft, delivery scams, etc.
✓ If using your inbox, anonymize/remove sender/recipient names
"""

# ============================================================================
# STEP 2: DATA FORMAT & STRUCTURE
# ============================================================================

import csv
import json
from pathlib import Path


def derive_subject_from_body(body, max_subject_len=120):
    """Create a fallback subject from body text when subject is missing."""
    text = (body or "").strip()
    if not text:
        return ""

    # Prefer the first non-empty line as subject when available.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[0][:max_subject_len]

    return text[:max_subject_len]


def normalize_csv_row(row):
    """Normalize incoming CSV rows to subject/body/label schema.

    Supports both:
    - subject, body, label
    - body, label (subject auto-derived from body)
    """
    normalized = {}
    for key, value in row.items():
        clean_key = str(key or "").strip().lower().lstrip("\ufeff")
        normalized[clean_key] = value

    subject = (normalized.get("subject", "") or "").strip()
    body = (normalized.get("body", "") or "").strip()
    label = (normalized.get("label", "") or "").strip()

    if not subject and body:
        subject = derive_subject_from_body(body)

    return subject, body, label

# REQUIRED CSV FORMAT:
"""
subject,body,label
"subject line 1","body text 1",0
"Account verification required","Click here to confirm your PayPal account: https://paypal-verify.com",1
"Team meeting scheduled","Hi everyone, our standup is at 10am tomorrow in Conf Room B",0

Rules:
- Column 1: subject (required, can be empty string if not available)
- Column 2: body (required, full email body text)
- Column 3: label (required, 0=legitimate, 1=phishing)
- Use CSV format (comma-delimited, quoted strings with commas/newlines)
- UTF-8 encoding
- One email per row
"""

# Example: Creating a correctly formatted CSV
def create_example_csv():
    """Show how to create a properly formatted CSV for training."""
    
    example_data = [
        {
            "subject": "Q1 Planning Session",
            "body": "Let's meet tomorrow at 2pm to discuss Q1 projects and deliverables.",
            "label": 0  # legitimate
        },
        {
            "subject": "ACTION REQUIRED: Verify Your Account",
            "body": "URGENT: Your account has been compromised. Click here immediately to verify your identity: https://secure-verify-bank.com/login or your account will be suspended.",
            "label": 1  # phishing
        },
        {
            "subject": "Your order has shipped",
            "body": "Your order #12345 has been dispatched and will arrive in 3-5 business days. Track it here: https://example-tracking.com/orders/12345",
            "label": 0  # legitimate
        },
    ]
    
    # Write to CSV
    output_file = "example_training_data.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["subject", "body", "label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(example_data)
    
    print(f"✓ Created example CSV: {output_file}")

# ============================================================================
# STEP 3: CLEANING & PREPARATION
# ============================================================================

def clean_email_text(text):
    """Remove PII and formatting issues from email text."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove common email headers (if present)
    lines = text.split("\n")
    cleaned_lines = [
        line for line in lines
        if not any(header in line for header in ["From:", "To:", "Date:", "Subject:"])
    ]
    text = "\n".join(cleaned_lines)
    
    return text.strip()

def validate_email_record(subject, body, label):
    """Validate a single email record."""
    errors = []
    
    # Check label
    try:
        label_int = int(label)
        if label_int not in [0, 1]:
            errors.append(f"Invalid label '{label}' (must be 0 or 1)")
    except (TypeError, ValueError):
        errors.append(f"Invalid label '{label}' (must be 0 or 1)")
    
    # Check body (required and non-empty)
    if not body or not str(body).strip():
        errors.append("Body is empty")
    
    # Check minimum length
    combined_text = f"{subject or ''} {body or ''}"
    if len(combined_text.strip()) < 10:
        errors.append(f"Email too short ({len(combined_text)} chars, min 10)")
    
    # Check maximum length (avoid outliers)
    if len(combined_text) > 10000:
        errors.append(f"Email too long ({len(combined_text)} chars, max 10000)")
    
    return errors

# ============================================================================
# STEP 4: DATA VALIDATION & BALANCE
# ============================================================================

def analyze_csv_dataset(csv_file):
    """Analyze a CSV dataset for quality and balance."""
    print(f"\n{'='*70}")
    print(f"DATASET ANALYSIS: {csv_file}")
    print(f"{'='*70}\n")
    
    if not Path(csv_file).exists():
        print(f"✗ File not found: {csv_file}")
        return
    
    legitimate_count = 0
    phishing_count = 0
    errors = []
    records = []
    
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, 1):
            subject, body, label = normalize_csv_row(row)
            
            # Validate
            record_errors = validate_email_record(subject, body, label)
            if record_errors:
                errors.append((idx, record_errors))
            
            # Count
            try:
                label_int = int(label)
                if label_int == 0:
                    legitimate_count += 1
                elif label_int == 1:
                    phishing_count += 1
                records.append(label_int)
            except ValueError:
                errors.append((idx, [f"Cannot parse label as integer: '{label}'"]))
    
    total = legitimate_count + phishing_count
    
    # Print summary
    print(f"Total Records: {total}")
    if total > 0:
        print(f"  ✓ Legitimate: {legitimate_count} ({legitimate_count/total*100:.1f}%)")
        print(f"  ✓ Phishing: {phishing_count} ({phishing_count/total*100:.1f}%)")
    else:
        print("  ✓ Legitimate: 0 (0.0%)")
        print("  ✓ Phishing: 0 (0.0%)")
    
    # Check balance
    imbalance_ratio = max(legitimate_count, phishing_count) / min(legitimate_count, phishing_count) if min(legitimate_count, phishing_count) > 0 else float('inf')
    if imbalance_ratio < 1.5:
        print(f"\n✓ Data is well balanced (ratio: {imbalance_ratio:.2f})")
    elif imbalance_ratio < 2.0:
        print(f"\n⚠ Data is slightly imbalanced (ratio: {imbalance_ratio:.2f})")
    else:
        print(f"\n✗ Data is heavily imbalanced (ratio: {imbalance_ratio:.2f})")
        print(f"  Recommendation: Collect more of the underrepresented class")
    
    # Print errors
    if errors:
        print(f"\n✗ Found {len(errors)} validation errors:")
        for row_num, error_list in errors[:10]:  # Show first 10
            print(f"  Row {row_num}: {', '.join(error_list)}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    else:
        print(f"\n✓ No validation errors found")
    
    print(f"\n{'='*70}\n")

# ============================================================================
# STEP 5: COMBINING DATASETS
# ============================================================================

def merge_csv_files(existing_csv, new_csv, output_csv):
    """Merge existing training data with new collected data."""
    print(f"Merging datasets...")
    print(f"  Existing: {existing_csv}")
    print(f"  New: {new_csv}")
    print(f"  Output: {output_csv}")
    
    records = []
    
    # Read existing
    if Path(existing_csv).exists():
        with open(existing_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subject, body, label = normalize_csv_row(row)
                records.append({"subject": subject, "body": body, "label": label})
    
    # Read new
    if Path(new_csv).exists():
        with open(new_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subject, body, label = normalize_csv_row(row)
                records.append({"subject": subject, "body": body, "label": label})
    
    # Write merged
    if records:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["subject", "body", "label"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        
        print(f"\n✓ Merged {len(records)} records into {output_csv}")
        analyze_csv_dataset(output_csv)

# ============================================================================
# STEP 6: DATA PREPARATION WORKFLOW
# ============================================================================

"""
RECOMMENDED WORKFLOW:

1. COLLECT DATA (1-2 weeks)
   ├─ Download phishing samples from PhishTank
   ├─ Extract corporate emails (anonymized) from your system
   ├─ Combine with Kaggle/GitHub datasets
   └─ Goal: 5,000-10,000 new examples

2. FORMAT DATA
   ├─ Create CSV with columns: subject, body, label
   ├─ Ensure label=0 for legitimate, 1 for phishing
   ├─ Apply clean_email_text() to remove PII
   ├─ Save as "new_training_data.csv"
   └─ Run validate & analyze

3. VALIDATE QUALITY
   ├─ Check: No errors in parsing
   ├─ Check: Minimum 10 chars, maximum 10k chars
   ├─ Check: Balance ratio < 2.0 (ideally 1.0-1.5)
   ├─ Check: Minimum 1,000 examples per class
   └─ Fix or remove problematic rows

4. MERGE WITH EXISTING
   ├─ Combine with enron_phishing_and_legitimate_email_dataset.csv
   ├─ Shuffle the merged dataset
   ├─ Save as "combined_training_dataset.csv"
   └─ Validate again

5. PREPARE FOR TRAINING
   ├─ Copy combined CSV to training directory
   ├─ Run train_models_improved.py with new data
   └─ Monitor training progress

EXAMPLE COMMAND:
  python prepare_data.py --validate new_training_data.csv
  python prepare_data.py --merge existing.csv new.csv combined.csv
"""

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("DATA PREPARATION GUIDE - USAGE EXAMPLES")
    print("="*80)
    
    print("\n1. VALIDATE YOUR DATA:")
    print("   analyze_csv_dataset('your_data.csv')")
    
    print("\n2. CLEAN EMAIL TEXT:")
    print("   cleaned = clean_email_text(raw_email)")
    
    print("\n3. VALIDATE SINGLE RECORD:")
    print("   errors = validate_email_record(subject, body, label)")
    
    print("\n4. CREATE EXAMPLE CSV:")
    print("   create_example_csv()  # Creates 'example_training_data.csv'")
    
    print("\n5. MERGE DATASETS:")
    print("   merge_csv_files('existing.csv', 'new.csv', 'combined.csv')")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("""1. Collect or download new email data from the sources listed above
2. Format as CSV with columns: subject, body, label
3. Put new data in: datasets/training_data/new_training_data.csv
4. Run: python prepare_data.py validate new_training_data.csv
5. If validation passes, merge and prepare for training
6. Once ready, training will take 30-60 min depending on data size
    """)
