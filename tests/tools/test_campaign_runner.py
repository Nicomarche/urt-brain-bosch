from tools.dev.campaign_runner import (
    RouteCommandResult,
    _compute_metrics,
    _split_polyline_on_jumps,
)


def test_compute_metrics_trims_ground_truth_to_trace_start() -> None:
    metrics = _compute_metrics(
        expected_route={
            "waypoints": [
                [5.0, -2.0, 0.0],
                [6.0, -2.0, 0.0],
            ]
        },
        brain_events=[],
        gt_events=[
            {
                "thread": "ground_truth",
                "event": "gz_pose",
                "ts": 9.0,
                "brain_map_x": -6.0,
                "brain_map_y": 7.0,
            },
            {
                "thread": "ground_truth",
                "event": "gz_pose",
                "ts": 10.1,
                "brain_map_x": 5.0,
                "brain_map_y": -2.0,
            },
            {
                "thread": "ground_truth",
                "event": "gz_pose",
                "ts": 10.2,
                "brain_map_x": 5.2,
                "brain_map_y": -2.0,
            },
        ],
        auto_start_ts=11.0,
        trace_start_ts=10.0,
        max_p90_m=0.35,
        max_max_m=0.6,
        max_lane_offset_p90_m=0.12,
        max_lane_offset_max_m=0.20,
        route_result=RouteCommandResult(completed=True),
    )

    assert metrics["ground_truth"]["trace_start_ts"] == 10.0
    assert metrics["ground_truth"]["actual_path"] == [[5.0, -2.0], [5.2, -2.0]]


def test_split_polyline_on_jumps_breaks_teleport_segments() -> None:
    segments = _split_polyline_on_jumps(
        [
            (0.0, 0.0),
            (0.1, 0.0),
            (4.0, 4.0),
            (4.1, 4.0),
        ],
        max_jump_m=0.6,
    )

    assert segments == [[(0.0, 0.0), (0.1, 0.0)], [(4.0, 4.0), (4.1, 4.0)]]
