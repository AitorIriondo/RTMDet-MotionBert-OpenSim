"""
Fix for Pose2Sim v0.10.33+ Markers_Coco17.xml corruption bug.

Pose2Sim commit 6773d6c (Aug 15, 2025) accidentally corrupted the RHip and
LKnee marker positions in Markers_Coco17.xml while fixing head marker
placements. This causes wildly wrong OpenSim model scaling:
  - femur scale ~2.2x (should be ~1.1x)
  - pelvis scale ~0.84x (should be ~1.35x)

This script detects and fixes the corruption by restoring the correct
symmetric marker positions from v0.10.32.

Usage:
    python fix_pose2sim.py           # Auto-detect and fix
    python fix_pose2sim.py --check   # Check only, don't fix
"""

import argparse
import shutil
import sys
from pathlib import Path


# Correct marker positions (from Pose2Sim v0.10.32, commit a9e1403)
# RHip must mirror LHip: same X,Y, opposite Z sign
CORRECT_RHIP = "-0.06392744445800781 -0.08134311294555664 0.10540640790244724"
# LKnee must mirror RKnee: same X,Y, opposite Z sign
CORRECT_LKNEE = "-0.005410484544778264 -0.3861321573680546 -0.005110696942696956"

# Known corrupted values (from commit 6773d6c onward)
CORRUPTED_VALUE = "-0.00541048 -0.397886 -0.000611877"


def find_pose2sim_markers():
    """Find the Markers_Coco17.xml file in the Pose2Sim installation."""
    try:
        import Pose2Sim
        pose2sim_dir = Path(Pose2Sim.__file__).parent
    except ImportError:
        # Try common conda paths
        for base in [
            Path(sys.prefix),
            Path.home() / "AppData/Local/anaconda3/envs/Pose2Sim",
            Path("C:/ProgramData/anaconda3/envs/Pose2Sim"),
        ]:
            pose2sim_dir = base / "Lib/site-packages/Pose2Sim"
            if pose2sim_dir.exists():
                break
        else:
            return None

    markers_file = pose2sim_dir / "OpenSim_Setup" / "Markers_Coco17.xml"
    return markers_file if markers_file.exists() else None


def check_corruption(markers_file):
    """Check if Markers_Coco17.xml has the known corruption."""
    content = markers_file.read_text(encoding="utf-8")

    issues = []

    # Check RHip: should be symmetric with LHip
    if CORRUPTED_VALUE in content:
        # Count occurrences - corrupted file has it twice (RHip + LKnee)
        count = content.count(CORRUPTED_VALUE)
        if count >= 1:
            issues.append(f"Found corrupted coordinates ({count} occurrence(s))")

    # More specific checks
    lines = content.split("\n")
    in_rhip = False
    in_lknee = False
    for line in lines:
        if 'name="RHip"' in line:
            in_rhip = True
        elif 'name="LKnee"' in line:
            in_lknee = True
        elif "<location>" in line and in_rhip:
            loc = line.strip().replace("<location>", "").replace("</location>", "").strip()
            if loc == CORRUPTED_VALUE:
                issues.append(f"RHip has corrupted position: {loc}")
            in_rhip = False
        elif "<location>" in line and in_lknee:
            loc = line.strip().replace("<location>", "").replace("</location>", "").strip()
            if loc == CORRUPTED_VALUE:
                issues.append(f"LKnee has corrupted position: {loc}")
            in_lknee = False

    return issues


def fix_corruption(markers_file, backup=True):
    """Fix the corrupted marker positions."""
    content = markers_file.read_text(encoding="utf-8")

    if backup:
        backup_path = markers_file.with_suffix(".xml.bak")
        shutil.copy2(markers_file, backup_path)
        print(f"  Backup saved: {backup_path}")

    # Fix RHip: replace the corrupted value in the RHip marker block
    # We need to be careful since the corrupted value appears twice
    lines = content.split("\n")
    fixed_lines = []
    in_rhip = False
    in_lknee = False
    fixes_applied = 0

    for line in lines:
        if 'name="RHip"' in line:
            in_rhip = True
        elif 'name="LKnee"' in line:
            in_lknee = True

        if "<location>" in line and in_rhip and CORRUPTED_VALUE in line:
            line = line.replace(CORRUPTED_VALUE, CORRECT_RHIP)
            fixes_applied += 1
            in_rhip = False
        elif "<location>" in line and in_lknee and CORRUPTED_VALUE in line:
            line = line.replace(CORRUPTED_VALUE, CORRECT_LKNEE)
            fixes_applied += 1
            in_lknee = False
        elif "<location>" in line:
            in_rhip = False
            in_lknee = False

        fixed_lines.append(line)

    markers_file.write_text("\n".join(fixed_lines), encoding="utf-8")
    return fixes_applied


def apply_fix_from_project(markers_file):
    """Apply fix by copying the known-good file from the project."""
    project_file = Path(__file__).parent / "opensim_setup" / "Markers_Coco17.xml"
    if project_file.exists():
        backup_path = markers_file.with_suffix(".xml.bak")
        shutil.copy2(markers_file, backup_path)
        shutil.copy2(project_file, markers_file)
        print(f"  Replaced with project's fixed version")
        print(f"  Backup saved: {backup_path}")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Fix Pose2Sim Markers_Coco17.xml corruption")
    parser.add_argument("--check", action="store_true", help="Check only, don't fix")
    args = parser.parse_args()

    print("Pose2Sim Markers_Coco17.xml Bug Fix")
    print("=" * 50)

    markers_file = find_pose2sim_markers()
    if markers_file is None:
        print("ERROR: Could not find Pose2Sim Markers_Coco17.xml")
        print("Make sure Pose2Sim is installed in the current environment.")
        sys.exit(1)

    print(f"Found: {markers_file}")

    issues = check_corruption(markers_file)
    if not issues:
        print("OK: No corruption detected. Markers_Coco17.xml is correct.")
        sys.exit(0)

    print(f"\nCorruption detected ({len(issues)} issues):")
    for issue in issues:
        print(f"  - {issue}")

    if args.check:
        print("\nRun without --check to apply the fix.")
        sys.exit(1)

    print("\nApplying fix...")
    if apply_fix_from_project(markers_file):
        print("FIX APPLIED successfully (from project file).")
    else:
        fixes = fix_corruption(markers_file)
        print(f"FIX APPLIED successfully ({fixes} corrections).")

    # Verify
    verify_issues = check_corruption(markers_file)
    if verify_issues:
        print(f"WARNING: {len(verify_issues)} issues remain after fix!")
        sys.exit(1)
    else:
        print("Verification: OK - all markers are now correct.")


if __name__ == "__main__":
    main()
