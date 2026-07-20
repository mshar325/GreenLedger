"""Invariants for the shared pipeline: the property-type filter must survive the 2023
label change, the band->tier map must be total and ordered, and single-input encoding
must align to the trained feature columns."""
import numpy as np
import pandas as pd
import pytest

from greenledger.pipeline import (SMALL_BIZ_PATTERN, BAND_TO_TIER, LABELS, FLOOR_AREA_CAP_M2,
                                    PROXY_NUM, PROXY_CAT, _property_group, build_feature_matrix,
                                    encode_single_input)

# The bug that broke the migration: labels changed wording in 2023. Both eras must match.
PRE_2023 = ["A1/A2 Retail and Financial/Professional services",
            "B1 Offices and Workshop businesses",
            "A3/A4/A5 Restaurant and Cafes/Drinking Establishments and Hot Food takeaways"]
POST_2023 = ["Retail/Financial and Professional Services",
             "Offices and Workshop Businesses",
             "Restaurants and Cafes/Drinking Establishments/Takeaways"]
NON_SMALLBIZ = ["B8 Storage or Distribution", "C1 Hotels",
                "D1 Non-residential Institutions - Education"]


@pytest.mark.parametrize("label", PRE_2023 + POST_2023)
def test_smallbiz_pattern_matches_both_label_eras(label):
    assert SMALL_BIZ_PATTERN.search(label), f"filter should match {label!r}"


@pytest.mark.parametrize("label", NON_SMALLBIZ)
def test_smallbiz_pattern_excludes_non_smallbiz(label):
    assert not SMALL_BIZ_PATTERN.search(label), f"filter should NOT match {label!r}"


def test_band_to_tier_is_total_and_ordered():
    for band in ["A+", "A", "B", "C", "D", "E", "F", "G"]:
        assert band in BAND_TO_TIER
    assert set(BAND_TO_TIER.values()) == set(LABELS)
    # lower (better) bands map to lower-risk tiers
    order = {"Low": 0, "Medium": 1, "High": 2}
    seq = [order[BAND_TO_TIER[b]] for b in ["A+", "A", "B", "C", "D", "E", "F", "G"]]
    assert seq == sorted(seq), "tier severity must be monotone in band"


def test_property_group_maps_to_three_known_groups():
    groups = {_property_group(l) for l in PRE_2023 + POST_2023}
    assert groups <= {"Retail/Financial/Professional", "Office/Workshop", "Restaurant/Cafe"}
    assert _property_group("Restaurants and Cafes/Drinking Establishments/Takeaways") == "Restaurant/Cafe"
    assert _property_group("Offices and Workshop Businesses") == "Office/Workshop"


def test_floor_area_cap_is_small_business_scale():
    assert 100 <= FLOOR_AREA_CAP_M2 <= 1000  # small commercial, well below MEES's 1000m2 large-building line


def _toy_frame():
    return pd.DataFrame({
        "property_type_group": ["Office/Workshop", "Restaurant/Cafe"],
        "main_heating_fuel": ["Natural Gas", "Oil"],
        "aircon_present": ["Yes", "No"],
        "building_environment": ["Air Conditioning", "Heating and Natural Ventilation"],
        "uk_region": ["London", "Wales"],
        "log_floor_area": np.log([120.0, 80.0]),
        "lodgement_year": [2024, 2025],
    })


def test_encode_single_input_aligns_to_feature_columns():
    train = build_feature_matrix(_toy_frame(), PROXY_NUM, PROXY_CAT)
    cols = list(train.columns)
    row = encode_single_input({
        "property_type_group": "Office/Workshop", "uk_region": "London", "floor_area": 120,
        "lodgement_year": 2024, "main_heating_fuel": "Natural Gas", "aircon_present": "Yes",
        "building_environment": "Air Conditioning",
    }, cols)
    assert list(row.columns) == cols          # exact alignment, no extra/missing columns
    assert row.shape == (1, len(cols))
    assert not row.isna().any().any()          # unknown categories become 0, never NaN


def test_encode_handles_unseen_category_gracefully():
    train = build_feature_matrix(_toy_frame(), PROXY_NUM, PROXY_CAT)
    cols = list(train.columns)
    row = encode_single_input({
        "property_type_group": "Office/Workshop", "uk_region": "Scotland",  # not in toy training
        "floor_area": 120, "lodgement_year": 2024, "main_heating_fuel": "Hydrogen",  # unseen fuel
        "aircon_present": "Yes", "building_environment": "Air Conditioning",
    }, cols)
    assert list(row.columns) == cols and row.shape == (1, len(cols))
