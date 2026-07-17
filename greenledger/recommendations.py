"""Rule-based recommendation engine. Every quantified claim below is a published range
from a government source (DOE / EPA / ENERGY STAR / GSA), cited inline — never a number
invented for a specific business. Where no reliable published percentage exists for a
practice, the card says so instead of making one up.
"""

RECOMMENDATIONS = {
    "led_lighting": {
        "title": "Switch to LED lighting",
        "detail": "Commercial LED retrofits typically cut lighting electricity by up to "
                   "50% compared with fluorescent fixtures, and up to 85% when paired with "
                   "occupancy or daylight controls.",
        "source": "U.S. GSA / ENERGY STAR",
        "source_url": "https://www.energystar.gov/buildings/save-energy-commercial-buildings/ways-save/upgrade-lighting",
        "drivers": {"cool_bin", "WKHRS"},
    },
    "hvac_setback": {
        "title": "Use HVAC setback scheduling",
        "detail": "Setting the thermostat back 7-10°F for 8 hours a day (e.g. after "
                   "closing) typically saves up to 10% a year on heating and cooling.",
        "source": "U.S. Department of Energy",
        "source_url": "https://www.energy.gov/energysaver/programmable-thermostats",
        "drivers": {"ht1_bin", "cool_bin", "WKHRS"},
    },
    "insulation": {
        "title": "Improve insulation and air sealing",
        "detail": "Air sealing plus attic/wall insulation upgrades save an average of 15% "
                   "on heating and cooling costs.",
        "source": "U.S. EPA / ENERGY STAR",
        "source_url": "https://www.energystar.gov/saveathome/seal_insulate/why-seal-and-insulate",
        "drivers": {"YRCONC", "ht1_bin"},
    },
    "space_zoning": {
        "title": "Zone HVAC to occupied areas",
        "detail": "Conditioning only actively-used space (rather than the full footprint) "
                   "reduces waste in low-occupancy-density buildings. General efficiency "
                   "practice — savings vary by layout, so no single published percentage "
                   "applies broadly here.",
        "source": "U.S. DOE Better Buildings Solution Center",
        "source_url": "https://betterbuildingssolutioncenter.energy.gov/",
        "drivers": {"log_sqft", "NWKER"},
    },
}

# maps an engineered feature name -> the conceptual "driver" it represents
FEATURE_TO_DRIVER = {
    "cool_bin": "cool_bin", "ht1_bin": "ht1_bin", "WKHRS": "WKHRS",
    "YRCONC": "YRCONC", "log_sqft": "log_sqft", "NWKER": "NWKER",
}


def get_recommendations(top_driver_features: list[str], max_cards: int = 4) -> list[dict]:
    """top_driver_features: engineered feature names ranked by how much they pushed this
    specific prediction toward higher risk (from SHAP), most influential first."""
    drivers = {FEATURE_TO_DRIVER[f] for f in top_driver_features if f in FEATURE_TO_DRIVER}

    selected, seen = [], set()
    for rec_id, rec in RECOMMENDATIONS.items():
        if rec["drivers"] & drivers and rec_id not in seen:
            selected.append(rec)
            seen.add(rec_id)

    if not selected:  # fallback: broadly applicable, still cited, never fabricated
        selected = [RECOMMENDATIONS["led_lighting"], RECOMMENDATIONS["hvac_setback"]]

    return selected[:max_cards]
