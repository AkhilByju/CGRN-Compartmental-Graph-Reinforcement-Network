from experiments.nano_transformer.param_solver import solve_hidden_width_for_target


def test_solves_exact_affine_target():
    # fixed=100, coef=10 -> hidden=90 gives exactly 1000
    sol = solve_hidden_width_for_target(target=1000, fixed_params=100, coef_per_hidden_unit=10)
    assert sol.hidden == 90
    assert sol.parameter_count == 1000
    assert sol.deviation == 0


def test_picks_closest_integer_when_not_exact():
    # fixed=0, coef=3 -> target=1000 -> 1000/3=333.33 -> 333*3=999 (closer than 334*3=1002)
    sol = solve_hidden_width_for_target(target=1000, fixed_params=0, coef_per_hidden_unit=3)
    assert sol.hidden == 333
    assert sol.parameter_count == 999
    assert abs(sol.deviation) <= 3


def test_within_one_percent_for_realistic_transformer_scale():
    # Backbone-like magnitudes: fixed ~394k, coef ~1920 (SwiGLU, 5 layers, d_model=128)
    target = 1_000_000
    sol = solve_hidden_width_for_target(
        target=target, fixed_params=394_624, coef_per_hidden_unit=1920
    )
    assert abs(sol.relative_deviation) <= 0.01


def test_respects_min_hidden():
    sol = solve_hidden_width_for_target(
        target=1, fixed_params=0, coef_per_hidden_unit=1000, min_hidden=1
    )
    assert sol.hidden == 1
