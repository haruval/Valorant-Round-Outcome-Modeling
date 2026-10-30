"""Provisional cast-function tags, not observations of successful ability effects.

Names remain in replay evidence. The model receives shared functions instead of
agent-specific identities. Tags describe selected functions, not an exhaustive
simulation of each ability. Multi-tag counts must not be summed as unique casts.
"""

CATEGORIES = (
    "vision_block", "vision_impair", "recon", "detection_deployable", "damage",
    "mobility", "suppression", "detain", "weapon_ultimate", "other",
)
ABILITY_TAGS = {
    "alarmbot": ("detection_deployable",),
    "blade-storm": ("weapon_ultimate",),
    "cloudburst": ("vision_block",),
    "cyber-cage": ("vision_block",),
    "dark-cover": ("vision_block",),
    "flash-drive": ("vision_impair",),
    "frag-ment": ("damage",),
    "from-the-shadows": ("mobility",),
    "hunter's-fury": ("damage",),
    "lockdown": ("detain",),
    "nanoswarm": ("damage",),
    "neural-theft": ("recon",),
    "null-cmd": ("suppression",),
    "owl-drone": ("recon",),
    "paranoia": ("vision_impair",),
    "poison-cloud": ("vision_block",),
    "recon-bolt": ("recon",),
    "shock-bolt": ("damage",),
    "shrouded-step": ("mobility",),
    "snake-bite": ("damage",),
    "spycam": ("recon",),
    "tailwind": ("mobility",),
    "toxic-screen": ("vision_block",),
    "trapwire": ("detection_deployable",),
    "turret": ("detection_deployable", "damage"),
    "updraft": ("mobility",),
    "viper's-pit": ("vision_block",),
    "zero-point": ("recon", "suppression"),
}


def ability_tags(name):
    return ABILITY_TAGS.get(name, ("other",))


def taxonomy_contract():
    return {"version": "cast-functions-v1", "status": "provisional",
            "categories": list(CATEGORIES),
            "abilities": {name: list(tags) for name, tags in sorted(ABILITY_TAGS.items())},
            "semantics": "Cast function only; no confirmed contact, effect, reveal, or rotation"}
