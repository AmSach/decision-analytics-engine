"""
Mortgage Underwriting & Risk Scoring Engine
Calibrated credit decisioning compliant with RBI Master Directions on Housing Finance,
Ind AS 109 Expected Credit Loss (ECL), and Surrogate Exception Compensators.
Author: Aman Sachan (IIFT BBA in Business Analytics)
Rule: Zero-Self-Glazing, strict quantitative rigor, no em dashes or en dashes.
"""

from typing import Dict, Any, List, Optional
import numpy as np


class MortgageUnderwritingEngine:
    """
    Automated mortgage underwriting, surrogate exception compensation,
    and deal-saver restructuring engine for Home Loans and Loans Against Property (LAP).
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
        """
        Evaluates loan applicant through 3 operational layers:
        1. Statutory RBI Hard Compliance Check
        2. Calibrated Probability of Default (PD) & Ind AS 109 Staging
        3. Surrogate Exception Compensator & Deal-Saver Restructuring
        """
        req_loan = float(applicant.get('requested_loan_amount', 0.0))
        prop_val = float(applicant.get('property_valuation', 1.0))
        income = float(applicant.get('net_monthly_income', 1.0))
        existing_emis = float(applicant.get('existing_monthly_obligations', 0.0))
        cibil = int(applicant.get('cibil_score', 300))
        loan_type = applicant.get('loan_type', 'home_loan')
        tenure = int(applicant.get('tenure_months', 240))
        rate = float(applicant.get('interest_rate', 8.75))
        bounces = int(applicant.get('cheque_bounces', 0))
        circularity = float(applicant.get('circularity_score', 0.0))
        liquid_buffer = float(applicant.get('liquid_buffer_months', 4.0))
        aa_ratio = float(applicant.get('aa_income_ratio', 1.0))

        proposed_emi = self.calculate_emi(req_loan, rate, tenure)
        total_monthly_debt = existing_emis + proposed_emi
        foir = total_monthly_debt / max(1.0, income)
        ltv = req_loan / max(1.0, prop_val)
        allowed_ltv = self.get_allowed_ltv(loan_type, req_loan)
        foir_cap = self.max_foir_hnw if income >= 150000 else self.max_foir_standard

        # Calibrated Probability of Default (PD) formulation with non-linear interaction
        cibil_delta = (cibil - 720.0) / 100.0
        foir_delta = max(0.0, foir - 0.40)
        ltv_delta = max(0.0, ltv - 0.70)
        buffer_delta = max(0.0, liquid_buffer - 6.0) / 6.0

        logit_pd = (
            -4.50
            + 3.20 * foir_delta
            + 2.80 * ltv_delta
            - 1.20 * cibil_delta
            + 0.55 * bounces
            + 1.80 * circularity
            - 0.65 * buffer_delta
        )
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

        # -----------------------------------------------------
        # SURROGATE EXCEPTION COMPENSATOR ENGINE
        # -----------------------------------------------------
        foir_waiver = 0.0
        compensators_applied = []
        if ltv <= 0.55:
            foir_waiver += 0.075
            compensators_applied.append("Collateral Cushion (+7.5% FOIR waiver for LTV <= 55%)")
        elif ltv <= 0.65:
            foir_waiver += 0.040
            compensators_applied.append("Moderate Collateral (+4.0% FOIR waiver for LTV <= 65%)")

        if liquid_buffer >= 8.0:
            foir_waiver += 0.060
            compensators_applied.append("Liquidity Moat (+6.0% FOIR waiver for Savings >= 8 EMIs)")
        elif liquid_buffer >= 5.0:
            foir_waiver += 0.030
            compensators_applied.append("Moderate Liquidity (+3.0% FOIR waiver)")

        if bounces == 0 and aa_ratio >= 0.95:
            foir_waiver += 0.035
            compensators_applied.append("Clean Banking Discipline (+3.5% FOIR waiver)")

        adjusted_foir_cap = min(0.70, foir_cap + foir_waiver)

        reasons = []
        if ltv > allowed_ltv:
            reasons.append(f"LTV violation: {ltv*100:.1f}% exceeds statutory cap of {allowed_ltv*100:.1f}%")
        if foir > foir_cap:
            reasons.append(f"FOIR violation: {foir*100:.1f}% exceeds base maximum limit of {foir_cap*100:.1f}%")
        if cibil < self.cibil_hard_cutoff:
            reasons.append(f"CIBIL rejection: score {cibil} is below minimum {self.cibil_hard_cutoff}")
        if applicant.get('fraud_hit', False) or circularity >= 0.65:
            reasons.append("Fraud hit: duplicate, manipulated, or circular account documents detected")
        if not applicant.get('property_title_clear', True):
            reasons.append("Title defect: property lacks clear marketable title")

        counter_offer = None
        saved_by_exception = False

        # Decision Routing
        if applicant.get('fraud_hit', False) or circularity >= 0.65 or not applicant.get('property_title_clear', True) or cibil < 600 or ltv > (allowed_ltv + 0.08) or pd > 0.14 or bounces >= 3:
            decision = "HARD REJECT"
        elif ltv > allowed_ltv:
            max_allowed = np.round(prop_val * allowed_ltv, -4)
            decision = "REFER TO SENIOR UNDERWRITER"
            counter_offer = {
                'type': 'LTV_HAIRCUT',
                'max_sanction_amount': max_allowed,
                'action': f"Sanction reduced loan of INR {max_allowed:,.0f} to meet statutory LTV."
            }
        elif foir <= foir_cap and cibil >= 680 and pd < 0.040:
            decision = "INSTANT SANCTION"
        elif foir <= adjusted_foir_cap and cibil >= 650 and pd < 0.080 and len(compensators_applied) > 0:
            decision = "INSTANT SANCTION VIA EXCEPTION"
            saved_by_exception = True
        elif reasons or pd >= 0.040:
            decision = "REFER TO SENIOR UNDERWRITER"
            # Calculate Deal-Saver Options
            r = rate / (12.0 * 100.0)
            max_afford_emi = max(0.0, (income * adjusted_foir_cap) - existing_emis)
            max_feasible_loan = np.round(max_afford_emi * (((1 + r)**tenure) - 1) / (r * ((1 + r)**tenure)), -4)
            counter_offer = {
                'type': 'DEAL_SAVER_RESTRUCTURE',
                'max_feasible_loan': min(max_feasible_loan, prop_val * allowed_ltv),
                'action': f"Option 1: Haircut to INR {min(max_feasible_loan, prop_val * allowed_ltv):,.0f}."
            }
        else:
            decision = "INSTANT SANCTION"

        return {
            'decision': decision,
            'ltv_ratio': round(ltv, 4),
            'allowed_ltv': allowed_ltv,
            'foir_ratio': round(foir, 4),
            'foir_cap': foir_cap,
            'adjusted_foir_cap': round(adjusted_foir_cap, 4),
            'proposed_emi': round(proposed_emi, 2),
            'probability_of_default': round(pd, 4),
            'loss_given_default': round(lgd, 4),
            'expected_credit_loss': round(ecl, 2),
            'ecl_stage': stage,
            'rejection_reasons': reasons,
            'saved_by_exception': saved_by_exception,
            'compensators_applied': compensators_applied,
            'counter_offer': counter_offer
        }
