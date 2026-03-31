#!/usr/bin/env python3
"""
Test suite for BERT email classifier model.
Evaluates model performance on various email types.
"""

from ml_model import analyze_email_with_ai
import json

# Test cases: (email_text, expected_category, description)
TEST_CASES = [
    # Clear phishing attempts
    (
        "URGENT: Your bank account has been locked! Click here immediately to verify your identity at https://secure-bank-verify.com/login or your account will be suspended.",
        "PHISHING",
        "High-urgency account verification phishing"
    ),
    (
        "Action Required: Confirm your PayPal password now. Someone may have unauthorized access to your account. https://paypal-secure-update.com/verify",
        "PHISHING",
        "PayPal credential phishing"
    ),
    (
        "Your Amazon order cannot be delivered. Update shipping address: https://amazon-delivery-confirm.net/track?order=12345",
        "PHISHING",
        "Amazon delivery phishing"
    ),
    (
        "ALERT: Suspicious login detected. Verify it was you: https://google-security-verify.com/signin",
        "PHISHING",
        "Google security alert phishing"
    ),
    (
        "You have unclaimed tax refund of $1,200. Claim now: https://irs-refund-claim.com/submit",
        "PHISHING",
        "IRS tax refund phishing"
    ),
    
    # Clear legitimate emails
    (
        "Meeting tomorrow at 2pm in Conference Room B. Agenda: Q1 planning and budget review. Please bring your department reports.",
        "LEGITIMATE",
        "Internal meeting notice"
    ),
    (
        "Your order #89123 has shipped and will arrive in 3-5 business days. Track it here: https://amazon.com/orders",
        "LEGITIMATE",
        "Amazon legitimate shipping notice"
    ),
    (
        "Monthly team standup: Tuesday 10am, Zoom link will be sent separately. Let me know if you have conflicts.",
        "LEGITIMATE",
        "Team standup scheduling"
    ),
    (
        "Project status update: We've completed 70% of the Q1 deliverables. Full report attached. Great work team!",
        "LEGITIMATE",
        "Project status internal email"
    ),
    (
        "Reminder: Vacation request deadline is Friday. Submit via HR portal. Contact HR if you have questions.",
        "LEGITIMATE",
        "HR vacation notice"
    ),
    
    # Borderline/suspicious cases
    (
        "Your account requires immediate attention. Please log in to confirm recent activity. https://mybank.com/login",
        "SUSPICIOUS",
        "Suspicious urgency with legitimate domain"
    ),
    (
        "Click here to update your profile information - do not ignore this message.",
        "SUSPICIOUS",
        "Vague account update request"
    ),
    (
        "Verify your account within 24 hours to avoid suspension. https://example-verification.co.uk",
        "SUSPICIOUS",
        "Account verification threat with urgent deadline"
    ),
    (
        "We noticed unusual activity. For your security, please confirm your login details at our secure portal.",
        "SUSPICIOUS",
        "Security alert requesting credentials"
    ),
]

def run_tests():
    """Run all test cases and report results."""
    print("\n" + "="*80)
    print("BERT EMAIL CLASSIFIER - MODEL TEST SUITE")
    print("="*80 + "\n")
    
    results = {
        "PHISHING": {"total": 0, "correct": 0, "scores": []},
        "LEGITIMATE": {"total": 0, "correct": 0, "scores": []},
        "SUSPICIOUS": {"total": 0, "correct": 0, "scores": []},
    }
    
    for idx, (email_text, expected_cat, description) in enumerate(TEST_CASES, 1):
        print(f"Test {idx}: {description}")
        print("-" * 80)
        
        result = analyze_email_with_ai(email_text)
        actual_cat = result['classification']
        score = result['risk_score']
        
        # Map classification to category
        if "PHISHING" in actual_cat:
            actual_cat_simple = "PHISHING"
        elif "SUSPICIOUS" in actual_cat:
            actual_cat_simple = "SUSPICIOUS"
        else:
            actual_cat_simple = "LEGITIMATE"
        
        is_correct = actual_cat_simple == expected_cat
        results[expected_cat]["total"] += 1
        results[expected_cat]["scores"].append(score)
        if is_correct:
            results[expected_cat]["correct"] += 1
        
        status = "✓ PASS" if is_correct else "✗ FAIL"
        print(f"Expected: {expected_cat:12} | Got: {actual_cat_simple:12} | Score: {score}% | {status}")
        print(f"Details: {actual_cat}")
        if result['nlp_findings']:
            findings_str = " | ".join(result['nlp_findings'][:2])
            print(f"Findings: {findings_str}")
        print()
    
    # Summary statistics
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80 + "\n")
    
    total_tests = len(TEST_CASES)
    total_correct = sum(results[cat]["correct"] for cat in results)
    overall_accuracy = (total_correct / total_tests * 100) if total_tests > 0 else 0
    
    for category in ["PHISHING", "LEGITIMATE", "SUSPICIOUS"]:
        cat_data = results[category]
        if cat_data["total"] > 0:
            cat_accuracy = (cat_data["correct"] / cat_data["total"] * 100)
            avg_score = sum(cat_data["scores"]) / len(cat_data["scores"])
            min_score = min(cat_data["scores"])
            max_score = max(cat_data["scores"])
            print(f"{category} Emails:")
            print(f"  Accuracy: {cat_accuracy:.1f}% ({cat_data['correct']}/{cat_data['total']} correct)")
            print(f"  Risk Scores: avg={avg_score:.1f}%, min={min_score}%, max={max_score}%")
            print()
    
    print(f"Overall Accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_tests} correct)\n")
    
    # Recommendations
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80 + "\n")
    
    if overall_accuracy >= 90:
        print("✓ Model performs well. No immediate retraining needed.")
        print("  Continue monitoring with real-world samples.")
    elif overall_accuracy >= 75:
        print("⚠ Model shows acceptable performance but has room for improvement.")
        print("  Consider collecting misclassified samples for targeted retraining.")
    else:
        print("✗ Model accuracy is below acceptable threshold.")
        print("  Recommend collecting more diverse training data and retraining.")
    
    print()

if __name__ == "__main__":
    run_tests()
