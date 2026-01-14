# backend/app/rag/pesticide_mapping.py

PESTICIDE_MAP = {
    "rice_bacterialblight": {
        "pesticide": "Streptocycline + Copper oxychloride",
        "notes": "Spray in early morning, avoid rain. Maintain field sanitation."
    },
    "rice_blast": {
        "pesticide": "Tricyclazole",
        "notes": "Use recommended dose, avoid waterlogging."
    },
    "rice_brownspot": {
        "pesticide": "Mancozeb",
        "notes": "Ensure proper spacing and balanced nitrogen."
    },
    "rice_tungro": {
        "pesticide": "Vector control (insecticide on leafhoppers)",
        "notes": "Remove infected plants, control insect vectors."
    },
    "corn_blight": {
        "pesticide": "Carbendazim + Mancozeb",
        "notes": "Use crop rotation and disease-free seeds."
    },
    "corn_common_rust": {
        "pesticide": "Propiconazole",
        "notes": "Avoid dense planting, monitor early."
    },
    "corn_gray_leaf_spot": {
        "pesticide": "Strobilurin fungicides",
        "notes": "Use resistant hybrids where possible."
    },
    "corn_healthy": {
        "pesticide": "None",
        "notes": "No disease detected, keep monitoring the crop."
    },
    "cotton_bacterial_blight": {
        "pesticide": "Copper oxychloride",
        "notes": "Use resistant varieties, avoid overhead irrigation."
    },
    "cotton_aphids": {
        "pesticide": "Imidacloprid",
        "notes": "Spray in evening, protect beneficial insects."
    },
    "cotton_army_worm": {
        "pesticide": "Emamectin benzoate",
        "notes": "Monitor larvae early, avoid overuse."
    },
    "cotton_healthy": {
        "pesticide": "None",
        "notes": "No disease detected, maintain good agronomy."
    },
    "wheat_healthy": {
        "pesticide": "None",
        "notes": "No disease detected, keep monitoring."
    },
    "wheat_septoria": {
        "pesticide": "Tebuconazole",
        "notes": "Ensure good airflow, avoid excess nitrogen."
    },
    "wheat_stripe_rust": {
        "pesticide": "Propiconazole",
        "notes": "Spray at first appearance, use resistant varieties."
    },
}


def get_pesticide_info(class_name: str):
    return PESTICIDE_MAP.get(
        class_name,
        {
            "pesticide": "Not available",
            "notes": "No specific recommendation found for this disease.",
        },
    )
