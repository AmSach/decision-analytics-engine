"""
Mortgage Underwriting & Risk Scoring Engine
Calibrated credit decisioning compliant with RBI Master Directions on Housing Finance and Ind AS 109.
"""

from typing import Dict, Any, List
import numpy as np


class MortgageUnderwritingEngine:
    """
    Automated mortgage underwriting and credit decisioning engine
    for retail Home Loans and Loans Against Property (LAP).
    """

    def __init__(self):
        # RBI Prudential LTV Caps
        self.ltv_caps = {
            'home_loan_tier1': 0.90,  # <= 30 Lakhs
            'home_loan_tier2': 0.80,  # 30L to 75L
            'home_loan_tier3': 0.75,  # > 75L
            'lap': 0.60               # Loan Against Property
        }
        self.max_foir_standard = 0.50
        self.max_foir_hnw = 0.60
        self.cibil_hard_cutoff = 650

    def calculate_emi(self, principal: float, annual_rate: float, tenure_months: int) -> float:
        """Computes reducing-balance monthly installment."""
        if annual_rate <= 0 or tenure_months <= 0:
            return principal / max(1, tenure_months)
        r = annual_rate / (12.0 * 100.0)
        emi = principal * r * ((1.0 + r) ** tenure_months) / (((1.0 + r) ** tenure_months) - 1.0)
        return float(emi)

    def get_allowed_ltv(self, loan_type: str, requested_amount: float) -> float:
        """Returns statutory RBI LTV ceiling."""
        if loan_type == 'lap':
            return self.ltv_caps['lap']
        elif requested_amount <= 3000000:
            return self.ltv_caps['home_loan_tier1']
        elif requested_amount <= 7500000:
            return self.ltv_caps['home_loan_tier2']
        else:
            return self.ltv_caps['home_loan_tier3']

    def evaluate_application(self, applicant: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates loan eligibility, credit risk, and routing."""
        req_loan = float(applicant.get('requested_loan_amount', 0.0))
        prop_val = float(applicant.get('property_valuation', 1.0))
        income = float(applicant.get('net_monthly_income', 1.0))
        existing_emis = float(applicant.get('existing_monthly_obligations', 0.0))
        cibil = int(applicant.get('cibil_score', 300))
        loan_type = applicant.get('loan_type', 'home_loan')
        tenure = int(applicant.get('tenure_months', 240))
        rate = float(applicant.get('interest_rate', 8.75))

        proposed_emi = self.calculate_emi(req_loan, rate, tenure)
        total_monthly_debt = existing_emis + proposed_emi
        foir = total_monthly_debt / max(1.0, income)
        ltv = req_loan / max(1.0, prop_val)
        allowed_ltv = self.get_allowed_ltv(loan_type, req_loan)
        foir_cap = self.max_foir_hnw if income >= 150000 else self.max_foir_standard

        # Calibrated Probability of Default (PD) formulation
        cibil_delta = (cibil - 720.0) / 100.0
        foir_delta = max(0.0, foir - 0.40)
        ltv_delta = max(0.0, ltv - 0.70)
        bounces = int(applicant.get('cheque_bounces', 0))

        logit_pd = -4.50 + 3.20 * foir_delta + 2.80 * ltv_delta - 1.20 * cibil_delta + 0.55 * bounces
        pd = 1.0 / (1.0 + np.exp(-logit_pd))

        # Loss Given Default (LGD) with 25% distress haircut
        distress_recovery = prop_val * 0.75
        recovery = min(req_loan, distress_recovery)
        lgd = max(0.05, (req_loan - recovery) / req_loan) if req_loan > 0 else 0.05
        ecl = pd * req_loan * lgd

        # Ind AS 109 Staging
        if pd < 0.035 and cibil >= 700:
            stage = "Stage 1: Performing (12M ECL)"
        elif pd < 0.085:
            stage = "Stage 2: Underperforming / SICR (Lifetime ECL)"
        else:
            stage = "Stage 3: Credit Impaired / NPA (Specific Provision)"

        # Hard stops
        reasons = []
        if ltv > allowed_ltv:
            reasons.append(f"LTV violation: {ltv*100:.1f}% exceeds statutory cap of {allowed_ltv*100:.1f}%")
        if foir > foir_cap:
            reasons.append(f"FOIR violation: {foir*100:.1f}% exceeds maximum limit of {foir_cap*100:.1f}%")
        if cibil < self.cibil_hard_cutoff:
            reasons.append(f"CIBIL rejection: score {cibil} is below minimum {self.cibil_hard_cutoff}")
        if applicant.get('fraud_hit', False):
            reasons.append("Fraud hit: duplicate or manipulated documents detected")
        if not applicant.get('property_title_clear', True):
            reasons.append("Title defect: property lacks clear marketable title")

        if applicant.get('fraud_hit', False) or not applicant.get('property_title_clear', True) or cibil < 600 or ltv > (allowed_ltv + 0.08) or pd > 0.12:
            decision = "HARD REJECT"
        elif reasons or pd >= 0.040:
            decision = "REFER TO SENIOR UNDERWRITER"
        else:
            decision = "INSTANT SANCTION"

        return {
            'decision': decision,
            'ltv_ratio': round(ltv, 4),
            'allowed_ltv': allowed_ltv,
            'foir_ratio': round(foir, 4),
            'foir_cap': foir_cap,
            'proposed_emi': round(proposed_emi, 2),
            'probability_of_default': round(pd, 4),
            'loss_given_default': round(lgd, 4),
            'expected_credit_loss': round(ecl, 2),
            'ecl_stage': stage,
            'rejection_reasons': reasons
        }
