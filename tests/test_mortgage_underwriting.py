"""
Unit tests for automated mortgage underwriting engine with surrogate exception compensators.
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


def test_ltv_breach_amber_refer_with_counter_offer(engine):
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
    assert res['counter_offer'] is not None
    assert res['counter_offer']['type'] == 'LTV_HAIRCUT'


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
    assert res['ltv_ratio'] == 0.50


def test_surrogate_exception_rescues_borderline_foir(engine):
    # Borderline FOIR (58% > 50%), but low LTV (50%) and high liquid buffer (10 EMIs)
    app = {
        'requested_loan_amount': 3000000,
        'property_valuation': 6000000,  # LTV 50% <= 55% -> grants +7.5% waiver
        'net_monthly_income': 62000,
        'existing_monthly_obligations': 8000,
        'cibil_score': 720,
        'loan_type': 'home_loan',
        'liquid_buffer_months': 10.0,   # Savings >= 8 EMIs -> grants +6.0% waiver
        'cheque_bounces': 0,
        'aa_income_ratio': 1.00,
        'property_title_clear': True,
        'fraud_hit': False
    }
    res = engine.evaluate_application(app)
    assert res['decision'] == 'INSTANT SANCTION VIA EXCEPTION'
    assert res['saved_by_exception'] is True
    assert len(res['compensators_applied']) >= 2


def test_circularity_fraud_hard_stop(engine):
    app = {
        'requested_loan_amount': 4000000,
        'property_valuation': 8000000,
        'net_monthly_income': 100000,
        'existing_monthly_obligations': 10000,
        'cibil_score': 740,
        'loan_type': 'home_loan',
        'circularity_score': 0.75,  # High circular fund rotation
        'property_title_clear': True,
        'fraud_hit': False
    }
    res = engine.evaluate_application(app)
    assert res['decision'] == 'HARD REJECT'
    assert any('Fraud' in r for r in res['rejection_reasons'])
