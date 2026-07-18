"""Rule-based recommendation engine. Every quantified claim below is a published range
from a UK government or Carbon Trust source, cited inline -- never a number invented for
a specific business. Where no reliable published percentage exists for a practice, the
card says so instead of making one up. (v2: UK sources, replacing the original US
DOE/EPA citations from the CBECS-based version -- see PROCESS.md.)
"""

RECOMMENDATIONS = {
    "led_lighting": {
        "title": "Switch to LED lighting",
        "detail": "Lighting can account for up to 50% of the electricity bill in retail, "
                   "hospitality and office premises, and switching to LED can cut that "
                   "lighting cost by an estimated 80%.",
        "source": "The Carbon Trust — Lighting Overview Guide",
        "source_url": "https://www.carbontrust.com/our-work-and-impact/guides-reports-and-tools/lighting-overview-guide",
        "drivers": {"aircon_present", "building_environment"},
    },
    "thermostat": {
        "title": "Turn the thermostat down a notch",
        "detail": "Lowering the average temperature by 1°C can cut heating bills by up to "
                   "8%, and turning the thermostat down generally can reduce heating bills "
                   "by as much as 10%.",
        "source": "GOV.UK — Energy Efficiency for Businesses",
        "source_url": "https://businessenergyefficiency.campaign.gov.uk/",
        "drivers": {"main_heating_fuel", "building_environment"},
    },
    "insulation": {
        "title": "Improve loft and wall insulation",
        "detail": "Loft insulation (100-150mm) can reduce heat loss through the roof by up "
                   "to 90%; cavity wall insulation can cut heat loss through the walls by "
                   "around two-thirds.",
        "source": "GOV.UK — SME Guide to Energy Efficiency",
        "source_url": "https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/417410/DECC_advice_guide.pdf",
        "drivers": {"main_heating_fuel"},
    },
    "space_zoning": {
        "title": "Zone heating/cooling to occupied areas",
        "detail": "Conditioning only actively-used space, rather than the full floor area, "
                   "reduces waste in larger or lower-occupancy-density premises. General "
                   "efficiency practice — savings vary by layout, so no single published "
                   "percentage applies broadly here.",
        "source": "The Carbon Trust — Better Business Guide to Energy Saving",
        "source_url": "https://www.carbontrust.com/sites/default/files/documents/resource/public/Better-Business-Guide.pdf",
        "drivers": {"log_floor_area"},
    },
}

# maps an engineered feature name -> the conceptual "driver" it represents
FEATURE_TO_DRIVER = {
    "aircon_present": "aircon_present",
    "building_environment": "building_environment",
    "main_heating_fuel": "main_heating_fuel",
    "log_floor_area": "log_floor_area",
}


def get_recommendations(top_driver_features: list, max_cards: int = 4) -> list:
    """top_driver_features: engineered feature names ranked by how much they pushed this
    specific prediction toward higher risk (from SHAP), most influential first."""
    drivers = set()
    for f in top_driver_features:
        for key, driver in FEATURE_TO_DRIVER.items():
            if f == key or f.startswith(key + "_"):  # one-hot columns are "col_value"
                drivers.add(driver)

    selected, seen = [], set()
    for rec_id, rec in RECOMMENDATIONS.items():
        if rec["drivers"] & drivers and rec_id not in seen:
            selected.append(rec)
            seen.add(rec_id)

    if not selected:  # fallback: broadly applicable, still cited, never fabricated
        selected = [RECOMMENDATIONS["led_lighting"], RECOMMENDATIONS["thermostat"]]

    return selected[:max_cards]
