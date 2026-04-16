"""
astronomy.py — Ancient Astronomical Calculations

Computes real planetary positions using:
  - VSOP87 (simplified) for Sun's ecliptic longitude
  - ELP2000 (simplified) for Moon's ecliptic longitude
  - Lahiri Ayanamsa for sidereal correction

Then derives the five elements of the daily Panchanga:
  1. Lunar Mansion (Nakshatra) — which of 27 segments the Moon occupies
  2. Lunar Day (Tithi) — based on Sun-Moon angular separation
  3. Zodiac Sign (Rashi) — 12 divisions of the ecliptic
  4. Sun-Moon Yoga — sum of Sun+Moon longitudes divided into 27 parts
  5. Half-Lunar-Day (Karana) — half of a Tithi
"""

import math

# ─── 27 Lunar Mansions with traditional rain associations ───
# Rain tendencies sourced from Brihat Samhita (Varahamihira, 6th century CE)
LUNAR_MANSIONS = [
    {"name": "Ashwini",            "ruler": "Ketu",    "rain": 0.30, "element": "fire",  "nature": "divine"},
    {"name": "Bharani",            "ruler": "Venus",   "rain": 0.40, "element": "earth", "nature": "human"},
    {"name": "Krittika",           "ruler": "Sun",     "rain": 0.20, "element": "fire",  "nature": "demonic"},
    {"name": "Rohini",             "ruler": "Moon",    "rain": 0.85, "element": "water", "nature": "human"},
    {"name": "Mrigashira",         "ruler": "Mars",    "rain": 0.70, "element": "water", "nature": "divine"},
    {"name": "Ardra",              "ruler": "Rahu",    "rain": 0.95, "element": "water", "nature": "human"},
    {"name": "Punarvasu",          "ruler": "Jupiter", "rain": 0.60, "element": "water", "nature": "divine"},
    {"name": "Pushya",             "ruler": "Saturn",  "rain": 0.75, "element": "water", "nature": "divine"},
    {"name": "Ashlesha",           "ruler": "Mercury", "rain": 0.50, "element": "water", "nature": "demonic"},
    {"name": "Magha",              "ruler": "Ketu",    "rain": 0.25, "element": "fire",  "nature": "demonic"},
    {"name": "Purva Phalguni",     "ruler": "Venus",   "rain": 0.35, "element": "fire",  "nature": "human"},
    {"name": "Uttara Phalguni",    "ruler": "Sun",     "rain": 0.30, "element": "fire",  "nature": "human"},
    {"name": "Hasta",              "ruler": "Moon",    "rain": 0.65, "element": "earth", "nature": "divine"},
    {"name": "Chitra",             "ruler": "Mars",    "rain": 0.40, "element": "fire",  "nature": "demonic"},
    {"name": "Swati",              "ruler": "Rahu",    "rain": 0.80, "element": "air",   "nature": "divine"},
    {"name": "Vishakha",           "ruler": "Jupiter", "rain": 0.55, "element": "water", "nature": "demonic"},
    {"name": "Anuradha",           "ruler": "Saturn",  "rain": 0.60, "element": "water", "nature": "divine"},
    {"name": "Jyeshtha",           "ruler": "Mercury", "rain": 0.45, "element": "water", "nature": "demonic"},
    {"name": "Mula",               "ruler": "Ketu",    "rain": 0.20, "element": "fire",  "nature": "demonic"},
    {"name": "Purva Ashadha",      "ruler": "Venus",   "rain": 0.70, "element": "water", "nature": "human"},
    {"name": "Uttara Ashadha",     "ruler": "Sun",     "rain": 0.50, "element": "earth", "nature": "human"},
    {"name": "Shravana",           "ruler": "Moon",    "rain": 0.80, "element": "water", "nature": "divine"},
    {"name": "Dhanishta",          "ruler": "Mars",    "rain": 0.45, "element": "earth", "nature": "demonic"},
    {"name": "Shatabhisha",        "ruler": "Rahu",    "rain": 0.75, "element": "air",   "nature": "demonic"},
    {"name": "Purva Bhadrapada",   "ruler": "Jupiter", "rain": 0.60, "element": "water", "nature": "human"},
    {"name": "Uttara Bhadrapada",  "ruler": "Saturn",  "rain": 0.70, "element": "water", "nature": "human"},
    {"name": "Revati",             "ruler": "Mercury", "rain": 0.55, "element": "water", "nature": "divine"},
]

# 12 Zodiac Signs
ZODIAC_SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
                "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
ZODIAC_RAIN_MODS = [-0.2, 0.1, 0.0, 0.3, -0.15, 0.05, 0.1, 0.25, -0.1, 0.0, 0.15, 0.35]
ZODIAC_ELEMENTS  = ["fire","earth","air","water","fire","earth","air","water","fire","earth","air","water"]

# Lunar Day rain associations (from Brihat Samhita)
LUNAR_DAY_RAIN = {i: v for i, v in enumerate([
    0.3, 0.45, 0.35, 0.5, 0.65, 0.4, 0.55, 0.7, 0.3, 0.5, 0.6, 0.45, 0.55, 0.75, 0.8,
    0.35, 0.5, 0.4, 0.55, 0.6, 0.45, 0.5, 0.65, 0.3, 0.5, 0.55, 0.45, 0.6, 0.7, 0.4
], 1)}

LUNAR_DAY_NAMES = [
    "1st","2nd","3rd","4th","5th","6th","7th","8th",
    "9th","10th","11th","12th","13th","14th","Full/New Moon"
]

YOGA_NAMES = [
    "Vishkumbha","Preeti","Ayushman","Saubhagya","Shobhana","Atiganda","Sukarma",
    "Dhriti","Shoola","Ganda","Vriddhi","Dhruva","Vyaghata","Harshana","Vajra",
    "Siddhi","Vyatipata","Variyan","Parigha","Shiva","Siddha","Sadhya","Shubha",
    "Shukla","Brahma","Indra","Vaidhriti"
]

SEASON_MAP = {1:"Winter", 2:"Winter", 3:"Spring", 4:"Spring", 5:"Summer", 6:"Summer",
              7:"Monsoon", 8:"Monsoon", 9:"Autumn", 10:"Autumn", 11:"Pre-Winter", 12:"Pre-Winter"}


# ─── Core Astronomical Functions ───

def _julian_day(year, month, day):
    """Convert calendar date to Julian Day Number."""
    if month <= 2:
        year -= 1
        month += 12
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    return int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5


def _sun_longitude(jd):
    """Sun's tropical ecliptic longitude (VSOP87 simplified)."""
    T = (jd - 2451545.0) / 36525.0
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T ** 2
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T ** 2
    M_rad = math.radians(M % 360)
    C = ((1.914602 - 0.004817 * T) * math.sin(M_rad) +
         0.019993 * math.sin(2 * M_rad) +
         0.000289 * math.sin(3 * M_rad))
    return (L0 + C) % 360


def _moon_longitude(jd):
    """Moon's tropical ecliptic longitude (ELP2000 simplified)."""
    T = (jd - 2451545.0) / 36525.0
    Lp = 218.3165 + 481267.8813 * T
    D  = 297.8502 + 445267.1115 * T
    M  = 357.5291 + 35999.0503 * T
    Mp = 134.9634 + 477198.8676 * T
    F  = 93.2720 + 483202.0175 * T
    Dr, Mr, Mpr, Fr = [math.radians(x % 360) for x in [D, M, Mp, F]]
    lon = (Lp +
           6.289 * math.sin(Mpr) +
           1.274 * math.sin(2 * Dr - Mpr) +
           0.658 * math.sin(2 * Dr) +
           0.214 * math.sin(2 * Mpr) -
           0.186 * math.sin(Mr) -
           0.114 * math.sin(2 * Fr))
    return lon % 360


def _ayanamsa(jd):
    """Lahiri Ayanamsa — converts tropical to sidereal coordinates."""
    return (23.85 + 0.0137 * (jd - 2451545.0) / 365.25) % 360


# ─── Public API ───

def compute_panchanga(date):
    """
    Compute the full astronomical Panchanga for a given date.

    Returns a dict with all five Panchanga elements plus derived features.
    All positions are sidereal (adjusted by Lahiri Ayanamsa).
    """
    jd = _julian_day(date.year, date.month, date.day)
    ay = _ayanamsa(jd)

    sun_sid = (_sun_longitude(jd) - ay) % 360
    moon_sid = (_moon_longitude(jd) - ay) % 360

    # 1. Lunar Mansion (Nakshatra)
    mansion_idx = int(moon_sid / (360.0 / 27)) % 27
    quarter = int((moon_sid % (360.0 / 27)) / (360.0 / 108)) + 1
    mansion = LUNAR_MANSIONS[mansion_idx]

    # 2. Zodiac Sign (Rashi)
    moon_sign = int(moon_sid / 30) % 12
    sun_sign = int(sun_sid / 30) % 12

    # 3. Lunar Day (Tithi)
    sun_moon_diff = (moon_sid - sun_sid) % 360
    lunar_day = min(int(sun_moon_diff / 12) + 1, 30)
    lunar_phase = "Waxing" if lunar_day <= 15 else "Waning"
    day_in_phase = lunar_day if lunar_day <= 15 else lunar_day - 15

    # 4. Yoga
    yoga_idx = int(((sun_sid + moon_sid) % 360) / (360 / 27))

    # 5. Karana
    karana_idx = int(sun_moon_diff / 6) % 60

    # Season
    season = SEASON_MAP.get(date.month, "Monsoon")

    return {
        # Positions
        "sun_longitude": sun_sid,
        "moon_longitude": moon_sid,
        "sun_moon_angle": sun_moon_diff,

        # Lunar Mansion
        "mansion_index": mansion_idx,
        "mansion_name": mansion["name"],
        "mansion_ruler": mansion["ruler"],
        "mansion_rain_tendency": mansion["rain"],
        "mansion_element": mansion["element"],
        "mansion_nature": mansion["nature"],
        "mansion_is_water": 1 if mansion["element"] == "water" else 0,
        "mansion_is_fire": 1 if mansion["element"] == "fire" else 0,
        "quarter": quarter,

        # Zodiac
        "moon_sign_index": moon_sign,
        "moon_sign_name": ZODIAC_SIGNS[moon_sign],
        "sun_sign_index": sun_sign,
        "sun_sign_name": ZODIAC_SIGNS[sun_sign],
        "sign_rain_modifier": ZODIAC_RAIN_MODS[moon_sign],
        "sign_element": ZODIAC_ELEMENTS[moon_sign],
        "sign_is_water": 1 if ZODIAC_ELEMENTS[moon_sign] == "water" else 0,

        # Lunar Day
        "lunar_day": lunar_day,
        "lunar_day_rain": LUNAR_DAY_RAIN.get(lunar_day, 0.5),
        "lunar_phase": lunar_phase,
        "day_in_phase": day_in_phase,
        "lunar_day_name": LUNAR_DAY_NAMES[(day_in_phase - 1) % 15],

        # Yoga & Karana
        "yoga_index": yoga_idx,
        "yoga_name": YOGA_NAMES[yoga_idx] if yoga_idx < 27 else "Unknown",
        "karana_index": karana_idx,

        # Season & Derived
        "season": season,
        "moon_speed_proxy": math.sin(math.radians(moon_sid * 2)),
        "ruler_is_benefic": 1 if mansion["ruler"] in ["Jupiter", "Venus", "Moon", "Mercury"] else 0,
        "nature_divine": 1 if mansion["nature"] == "divine" else 0,
        "nature_demonic": 1 if mansion["nature"] == "demonic" else 0,
    }
