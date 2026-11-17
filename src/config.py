"""Central configuration: indicators, assets, and regime-labelling rules."""
START = "1990-01-01"

# FRED series id -> transform ("yoy" = 12m % change, "lvl" = use as-is)
FRED = {
    "INDPRO": "yoy", "PAYEMS": "yoy", "UNRATE": "lvl",
    "CPIAUCSL": "yoy", "PCEPILFE": "yoy",
    "FEDFUNDS": "lvl", "GS2": "lvl", "GS10": "lvl", "T10Y2Y": "lvl",
    "BAA10Y": "lvl", "VIXCLS": "lvl",
    "UMCSENT": "lvl", "RSAFS": "yoy", "HOUST": "lvl", "M2SL": "yoy",
}
NAMES = {
    "INDPRO": "Industrial production", "PAYEMS": "Nonfarm payrolls", "UNRATE": "Unemployment rate",
    "CPIAUCSL": "CPI inflation", "PCEPILFE": "Core PCE", "FEDFUNDS": "Fed funds rate",
    "GS2": "2Y yield", "GS10": "10Y yield", "T10Y2Y": "10Y-2Y slope", "BAA10Y": "Baa credit spread",
    "VIXCLS": "VIX", "UMCSENT": "Consumer sentiment", "RSAFS": "Retail sales",
    "HOUST": "Housing starts", "M2SL": "M2 growth",
}

# Asset class -> Yahoo Finance ticker. Swap any of these for a longer-history proxy.
# NOTE: GSG (commodities) only starts in 2006, so commodity stats cover fewer months.
ASSETS = {"Equities": "^GSPC", "Bonds": "VBMFX", "Currencies": "DX-Y.NYB", "Commodities": "GSG"}

# Used only to ORDER clusters (R1 = strongest growth) and tag them; not used for fitting.
GROWTH_POS = ["INDPRO", "PAYEMS", "RSAFS"]
GROWTH_NEG = ["UNRATE"]
