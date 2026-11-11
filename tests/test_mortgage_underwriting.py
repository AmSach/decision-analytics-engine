"""
Unit tests for automated mortgage underwriting engine.
"""

import pytest
from src.mortgage_underwriting_engine import MortgageUnderwritingEngine


@pytest.fixture
def engine():
    return MortgageUnderwritingEngine()


def test_prime_instant_sanction(engine):
    app = {
        'requested_loan_amount': 4500000,
        'property_valuation': 6000000,  # LTV 75% <= 80%
        'net_monthly_income': 130000,
        'existing_monthly_obligations': 10000,
        'cibil_score': 780,
        'loan_type': 'home_loan',
        'property_title_clear': True,
        'fraud_hit': False
    }
    res = engine.evaluate_application(app)
    assert res['decision'] == 'INSTANT SANCTION'
    assert res['ltv_ratio'] == 0.75
    assert res['ecl_stage'].startswith('Stage 1')
    assert len(res['rejection_reasons']) == 0


def test_ltv_breach_amber_refer(engine):
    app = {
        'requested_loan_amount': 2850000,
        'property_valuation': 3000000,  # LTV 95% > 90%
        'net_monthly_income': 65000,
        'existing_monthly_obligations': 5000,
        'cibil_score': 720,
        'loan_type': 'home_loan',
        'property_title_clear': True,
        'fraud_hit': False
    }
    res = engine.evaluate_application(app)
    assert 'REFER' in res['decision'] or 'REJECT' in res['decision']
    assert any('LTV' in r for r in res['rejection_reasons'])


def test_lap_strict_ltv_cap(engine):
    # LAP LTV cap is 60%
    assert engine.get_allowed_ltv('lap', 5000000) == 0.60
    app = {
        'requested_loan_amount': 5000000,
        'property_valuation': 10000000,  # LTV 50% <= 60%
        'net_monthly_income': 180000,
        'existing_monthly_obligations': 20000,
        'cibil_score': 750,
        'loan_type': 'lap',
        'property_title_clear': True,
        'fraud_hit': False
    }
    res = engine.evaluate_application(app)
    assert res['decision'] == 'INSTANT SANCTION'
    assert res['allowed_ltv'] == 0.60


def test_fraud_hard_reject(engine):
    app = {
        'requested_loan_amount': 3000000,
        'property_valuation': 5000000,
        'net_monthly_income': 80000,
        'cibil_score': 760,
        'fraud_hit': True
    }
    res = engine.evaluate_application(app)
    assert res['decision'] == 'HARD REJECT'
