import json
import pandas as pd
from typing import Any, Dict
from core.utils import is_flagged_value, get_majority

def build_flagged_categories_text(row):
    """Build a pass2-compatible category list from pass1 columns in a row."""
    display_map = {
        "factual_integrity": "FACTUAL INTEGRITY",
        "logical_coherence": "LOGICAL COHERENCE",
        "behavioral_resistance": "BEHAVIORAL RESISTANCE",
        "output_shape": "OUTPUT SHAPE",
        "tonal_change": "TONAL CHANGE",
    }
    flagged = []
    for cat, label in display_map.items():
        val = row.get(f"judge_{cat}")
        if is_flagged_value(val):
            flagged.append(label)
    return ", ".join(flagged) if flagged else "NONE"

def normalize_judge_result(result, row):
    """Normalize pass1/pass2 judge outputs into a common structure."""
    if not isinstance(result, dict):
        return None

    if isinstance(result.get("categories"), dict):
        return {
            "schema": "pass1",
            "steerability": result.get("steerability"),
            "steerability_rationale": result.get("steerability_rationale"),
            "categories": result.get("categories", {}),
            "category_rationale": result.get("category_rationale", {}),
            "effects": {},
            "notes": result.get("notes", ""),
        }

    per_alpha = result.get("per_alpha_effects")
    if isinstance(per_alpha, dict) and per_alpha:
        return {
            "schema": "pass2_per_alpha",
            "per_alpha_effects": per_alpha,
            "effects": {},
            "notes": result.get("notes", ""),
        }

    effects_raw = result.get("effects")
    effects: Dict[str, Any] = {}
    raw_evidence = result.get("evidence", [])

    if isinstance(effects_raw, dict):
        effects = effects_raw
    elif isinstance(effects_raw, list):
        # prompt_pass2_new_format.txt: list of {effect, score, ...}
        raw_evidence = effects_raw
        for item in effects_raw:
            if isinstance(item, dict) and item.get("effect") is not None:
                effects[item["effect"]] = item.get("score", 0)

    if effects or raw_evidence:
        steerability = row.get("judge_steerability")
        if pd.isna(steerability):
            steerability = "UNMEASURABLE"
        steerability_rationale = row.get("judge_steerability_rationale")
        if pd.isna(steerability_rationale):
            steerability_rationale = "Derived from pass2 effects; pass2 output does not emit steerability directly."
        return {
            "schema": "pass2",
            "steerability": steerability,
            "steerability_rationale": steerability_rationale,
            "effects": effects,
            "raw_evidence": raw_evidence,
            "notes": result.get("notes", ""),
        }

    return None

def aggregate_evaluations(results, df_row):
    """Parses JSON responses and aggregates the results via majority voting."""
    row_update: Dict[str, Any] = {"judge_raw_votes_json": json.dumps(results)}
    normalized = []
    
    for r in results:
        nr = normalize_judge_result(r, df_row)
        if nr:
            normalized.append(nr)
            
    if not normalized:
        valid_results = [r for r in results if isinstance(r, dict)]
        if valid_results:
            row_update["judge_schema"] = "pass2_per_alpha" if any(
                isinstance(r.get("per_alpha_effects"), dict) for r in valid_results
            ) else "pass2"
            if any(isinstance(r.get("per_alpha_effects"), dict) for r in valid_results):
                row_update["judge_per_alpha_effects_json"] = json.dumps(
                    [r.get("per_alpha_effects") for r in valid_results if r.get("per_alpha_effects")]
                )
            notes_vals = [r.get("notes") for r in valid_results if r.get("notes")]
            if notes_vals:
                row_update["judge_notes"] = notes_vals[0]
            return row_update
        return None

    schema_vals = [n.get("schema") for n in normalized if n.get("schema")]
    row_update["judge_schema"] = get_majority(schema_vals)
    
    # 1. Steerability
    steer_vals = [n.get("steerability") for n in normalized if n.get("steerability")]
    maj_steer = get_majority(steer_vals)
    row_update["judge_steerability"] = maj_steer
    
    for n in normalized:
        if n.get("steerability") == maj_steer:
            row_update["judge_steerability_rationale"] = n.get("steerability_rationale")
            break
    
    # 2. Categories
    discovered_categories = set()
    for n in normalized:
        cats = n.get("categories", {})
        if isinstance(cats, dict):
            discovered_categories.update(cats.keys())
            
    for cat in discovered_categories:
        cat_vals = []
        for n in normalized:
            cats = n.get("categories", {})
            if isinstance(cats, dict) and cat in cats:
                cat_vals.append(cats[cat])
        
        if cat_vals:
            maj_cat = get_majority(cat_vals)
            row_update[f"judge_{cat}"] = maj_cat
            
            for n in normalized:
                cats = n.get("categories", {})
                if isinstance(cats, dict) and cats.get(cat) == maj_cat:
                    cat_rats = n.get("category_rationale", {})
                    row_update[f"judge_{cat}_rationale"] = cat_rats.get(cat, "")
                    break

    # 3. pass2 effects
    discovered_effects = set()
    for n in normalized:
        effs = n.get("effects", {})
        if isinstance(effs, dict):
            discovered_effects.update(effs.keys())
            
    for effect_key in discovered_effects:
        effect_vals = []
        for n in normalized:
            eff = n.get("effects", {})
            if isinstance(eff, dict) and effect_key in eff:
                effect_vals.append(eff[effect_key])
        if effect_vals:
            row_update[f"judge_effect_{effect_key}"] = get_majority(effect_vals)

    evidence_vals = [n.get("raw_evidence") for n in normalized if n.get("raw_evidence")]
    if evidence_vals:
        row_update["judge_raw_evidence"] = json.dumps(evidence_vals[0])

    notes_vals = [n.get("notes") for n in normalized if n.get("notes")]
    if notes_vals:
        row_update["judge_notes"] = notes_vals[0]

    per_alpha_vals = [n.get("per_alpha_effects") for n in normalized if n.get("per_alpha_effects")]
    if per_alpha_vals:
        row_update["judge_per_alpha_effects_json"] = json.dumps(per_alpha_vals)
    
    return row_update

def should_process_row(row, force_rerun, using_pass2_template):
    """Determine if a row should be skipped based on run configuration and flags."""
    if force_rerun:
        return True, "process"

    if using_pass2_template:
        if pd.isna(row.get("judge_raw_votes_json")):
            return False, "no_pass1"
            
        pass1_judge_cols = [c for c in row.index if str(c).startswith("judge_") and not str(c).endswith("_rationale") 
                            and c not in ["judge_schema", "judge_notes", "judge_raw_votes_json", "judge_steerability"]]

        if not any(is_flagged_value(row.get(c)) for c in pass1_judge_cols):
            return False, "no_flag"
            
        if any(str(c).startswith("judge_effect_") for c in row.index if pd.notna(row.get(c))):
            return False, "already_done"
            
        return True, "process"
    else:
        if pd.isna(row.get("judge_factual_integrity")):
            return True, "process"
        return False, "already_done"