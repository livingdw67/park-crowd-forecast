import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "databricks" / "src"))

from parks_lib.staffing import arrivals_by_hour, evaluate_plan, normalize_name, plan_staffing, shift_templates

RIDE = {"location": "Test Coaster", "kind": "ride", "daily_share": 0.6, "guests_per_crew_hour": 120, "min_crew": 6, "max_crew": 12}


def test_shift_templates_fit_inside_operating_hours():
    shifts = shift_templates(9, 21)
    assert all(9 <= start and end <= 21 for start, end in shifts)
    assert (13, 21) in shifts  # last full shift ends at close even off the 2-hour grid
    assert (17, 21) in shifts


def test_arrivals_sum_to_daily_guests():
    arrivals = arrivals_by_hour(10000, 9, 21)
    assert abs(sum(arrivals.values()) - 10000) < 1e-6
    assert max(arrivals, key=arrivals.get) in (13, 14)


def test_plan_respects_min_and_max_crew():
    plan = plan_staffing([RIDE], 50)
    assert plan.status == "optimal"
    assert all(6 <= h["crew"] <= 12 for h in plan.hourly)


def test_busier_day_means_longer_waits_for_the_same_plan():
    plan = plan_staffing([RIDE], 40)
    same = evaluate_plan(plan, [RIDE], 40)
    busier = evaluate_plan(plan, [RIDE], 70)
    assert same.wait_guest_hours == plan.wait_guest_hours
    assert busier.wait_guest_hours > plan.wait_guest_hours


def test_headcount_cap_is_respected():
    second = {**RIDE, "location": "Second Coaster"}
    free = plan_staffing([RIDE, second], 80)
    capped = plan_staffing([RIDE, second], 80, headcount=free.total_workers - 4)
    assert capped.total_workers <= free.total_workers - 4


def test_normalize_name_matches_queue_times_spelling():
    assert normalize_name("MEN IN BLACK™ Alien Attack!") == normalize_name("MEN IN BLACK Alien Attack")
    assert normalize_name("Mario Kart™: Bowser's Challenge") == "mario kart bowsers challenge"
