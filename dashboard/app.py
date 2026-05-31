"""Streamlit dashboard for RoboGenesis research memory."""

from __future__ import annotations

import argparse
import html
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.components.experiment_table import load_experiments
from dashboard.components.metric_charts import load_policies
from dashboard.components.run_review import format_score, format_score_delta, format_value, load_run_reviews
from dashboard.components.scenario_tree import load_scenarios


def run_dashboard(db_path: Path) -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit("Install dashboard extras first: pip install -e '.[dashboard]'") from exc

    st.set_page_config(page_title="RoboGenesis", layout="wide")
    _install_styles(st)
    st.title("RoboGenesis AutoResearch")

    if not db_path.exists():
        st.warning(f"No research DB found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    try:
        policies = load_policies(conn)
        experiments = load_experiments(conn)
        scenarios = load_scenarios(conn)
        run_reviews = load_run_reviews(conn, repo_root=REPO_ROOT)
    finally:
        conn.close()

    best = max(policies, key=lambda row: row.get("score", 0.0), default=None)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Policy Refs", _policy_reference_count(policies, run_reviews))
    col_b.metric("Experiments", len(experiments))
    col_c.metric("Scenarios", len(scenarios))

    review_tab, graph_tab, lineage_tab, tables_tab = st.tabs(["Run Review", "System Graph", "Lineage", "Tables"])
    with review_tab:
        _render_run_review(st, run_reviews)
    with graph_tab:
        _render_system_graph(st, run_reviews)
    with lineage_tab:
        if best:
            st.subheader("Current Best Policy")
            st.json(best)

        st.subheader("Experiment Leaderboard")
        st.dataframe(experiments, use_container_width=True, hide_index=True)

        st.subheader("Scenario Tree")
        st.dataframe(scenarios, use_container_width=True, hide_index=True)

        st.subheader("Policy Scores")
        if policies:
            st.dataframe(policies, use_container_width=True, hide_index=True)
        else:
            st.info("No scored policies have been ingested yet. Recorded runs still reference parent policies.")
    with tables_tab:
        _render_raw_tables(st, run_reviews, experiments, scenarios, policies)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="artifacts/research.db")
    args = parser.parse_args()
    run_dashboard(Path(args.db))


def _render_run_review(st: Any, reviews: list[dict[str, Any]]) -> None:
    st.subheader("Run Review")
    if not reviews:
        st.info("No AutoResearch runs are recorded yet.")
        return

    selected_index = st.selectbox(
        "Run",
        range(len(reviews)),
        index=_default_review_index(reviews),
        format_func=lambda index: _run_label(reviews[index]),
    )
    review = reviews[int(selected_index)]

    _render_run_header(st, review)
    render_col, rationale_col = st.columns([1.35, 1.0], gap="large")
    with render_col:
        _render_videos(st, review)
    with rationale_col:
        _render_rationale(st, review)

    _render_changes(st, review)
    _render_actions_and_tasks(st, review)


def _render_system_graph(st: Any, reviews: list[dict[str, Any]]) -> None:
    st.subheader("System Graph")
    if not reviews:
        st.info("No AutoResearch runs are recorded yet.")
        return

    selected_index = st.selectbox(
        "Graph Run",
        range(len(reviews)),
        index=_default_review_index(reviews),
        format_func=lambda index: _run_label(reviews[index]),
        key="system_graph_run",
    )
    review = reviews[int(selected_index)]

    st.graphviz_chart(_system_graph_dot(review), use_container_width=True)
    cols = st.columns(4)
    cols[0].metric("Run", _short_text(str(review["experiment_id"]), 28))
    cols[1].metric("Status", str(review.get("status") or "n/a"))
    cols[2].metric("Render Videos", len([video for video in review.get("run_videos") or [] if video.get("exists")]))
    cols[3].metric("Patch Changes", len(review.get("change_rows") or []))


def _render_run_header(st: Any, review: dict[str, Any]) -> None:
    st.markdown(
        "<div class='run-title'>"
        f"<span>{html.escape(str(review['experiment_id']))}</span>"
        f"<span class='status-pill'>{html.escape(str(review['status']))}</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    cols[0].metric("Score Before", format_score(review.get("score_before")))
    cols[1].metric("Score After", format_score(review.get("score_after")))
    cols[2].metric("Delta", format_score_delta(review.get("score_delta")))
    cols[3].metric("Changes", len(review.get("change_rows") or []))
    cols[4].metric("Scenarios", len(review.get("generated_scenarios") or []))
    details = [
        ("Parent Policy", review.get("parent_policy_id")),
        ("Created", review.get("created_at")),
        ("Task", review.get("task")),
        ("Modal Job", review.get("modal_job_id")),
    ]
    st.markdown(
        "<div class='detail-grid'>"
        + "".join(
            "<div class='detail-cell'>"
            f"<span>{html.escape(label)}</span><strong>{html.escape(str(value or 'n/a'))}</strong>"
            "</div>"
            for label, value in details
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def _render_videos(st: Any, review: dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown("#### Render")
        run_videos = review.get("run_videos") or []
        source_videos = review.get("source_videos") or []
        if run_videos:
            _render_video_group(st, "Run Output", run_videos)
        else:
            st.caption("No completed run render is available locally.")
        if source_videos:
            _render_video_group(st, "Source Render", source_videos)


def _render_video_group(st: Any, title: str, videos: list[dict[str, Any]]) -> None:
    st.markdown(f"**{title}**")
    existing = [video for video in videos if video.get("exists")]
    missing = [video for video in videos if not video.get("exists")]
    if not existing:
        st.caption("Referenced render files are not present on disk.")
    elif len(existing) == 1:
        _render_one_video(st, existing[0])
    else:
        tabs = st.tabs([_video_tab_label(video, index) for index, video in enumerate(existing, start=1)])
        for tab, video in zip(tabs, existing, strict=True):
            with tab:
                _render_one_video(st, video)
    if missing:
        with st.expander("Missing render references", expanded=False):
            st.dataframe(
                [{"label": video.get("label"), "path": video.get("path")} for video in missing],
                use_container_width=True,
                hide_index=True,
            )


def _render_one_video(st: Any, video: dict[str, Any]) -> None:
    st.video(str(video["path"]))
    st.caption(str(video["path"]))


def _render_rationale(st: Any, review: dict[str, Any]) -> None:
    rationale = review.get("rationale") or {}
    with st.container(border=True):
        st.markdown("#### Planner Rationale")
        primary_failure = rationale.get("primary_failure")
        if primary_failure:
            st.markdown(f"<span class='status-pill muted'>Failure: {html.escape(str(primary_failure))}</span>", unsafe_allow_html=True)
        for label, key in [
            ("Hypothesis", "hypothesis"),
            ("Expected Effect", "expected_effect"),
            ("Risk", "risk"),
            ("Rollback", "rollback"),
        ]:
            value = rationale.get(key)
            if value:
                st.markdown(f"**{label}**")
                st.write(value)

        likely_causes = rationale.get("likely_causes") or []
        directions = rationale.get("suggested_research_directions") or []
        if likely_causes:
            st.markdown("**Likely Causes**")
            for cause in likely_causes:
                st.write(f"- {cause}")
        if directions:
            st.markdown("**Next Research Directions**")
            for direction in directions:
                st.write(f"- {direction}")

        failure_report = review.get("failure_report") or {}
        evidence = failure_report.get("evidence") if isinstance(failure_report.get("evidence"), dict) else {}
        aggregate = evidence.get("aggregate") if isinstance(evidence.get("aggregate"), dict) else {}
        if aggregate:
            st.markdown("**Evidence Summary**")
            st.dataframe(
                [{"metric": key, "value": format_value(value)} for key, value in sorted(aggregate.items())],
                use_container_width=True,
                hide_index=True,
            )


def _render_changes(st: Any, review: dict[str, Any]) -> None:
    st.subheader("Section Changes")
    rows = review.get("change_rows") or []
    if not rows:
        st.info("No recorded config changes for this run.")
        return
    sections = review.get("changed_sections") or sorted({row["section"] for row in rows})
    tabs = st.tabs(sections)
    for tab, section in zip(tabs, sections, strict=True):
        with tab:
            section_rows = [
                {
                    "parameter": row["parameter"],
                    "old": row["old"],
                    "new": row["new"],
                    "file": row["file"],
                }
                for row in rows
                if row["section"] == section
            ]
            st.dataframe(section_rows, use_container_width=True, hide_index=True)


def _render_actions_and_tasks(st: Any, review: dict[str, Any]) -> None:
    st.subheader("Actions And Tasks")
    action_rows = _action_rows(review)
    st.dataframe(action_rows, use_container_width=True, hide_index=True)

    scenarios = review.get("generated_scenarios") or []
    if scenarios:
        with st.container(border=True):
            st.markdown("#### Generated Scenario Tasks")
            st.dataframe(_scenario_rows(scenarios), use_container_width=True, hide_index=True)

    training_surface = review.get("training_surface") or {}
    if training_surface:
        with st.expander("Training Surface", expanded=False):
            st.json(training_surface)

    paths = review.get("artifact_paths") or {}
    if any(paths.values()):
        with st.expander("Artifact Paths", expanded=False):
            st.json(paths)


def _render_raw_tables(
    st: Any,
    reviews: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    policies: list[dict[str, Any]],
) -> None:
    st.subheader("Runs")
    st.dataframe(
        [
            {
                "experiment_id": review["experiment_id"],
                "status": review["status"],
                "parent_policy_id": review.get("parent_policy_id"),
                "score_before": review.get("score_before"),
                "score_after": review.get("score_after"),
                "changes": len(review.get("change_rows") or []),
                "renders": len(review.get("videos") or []),
                "created_at": review.get("created_at"),
            }
            for review in reviews
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("Experiments")
    st.dataframe(experiments, use_container_width=True, hide_index=True)
    st.subheader("Scenarios")
    st.dataframe(scenarios, use_container_width=True, hide_index=True)
    st.subheader("Policies")
    if policies:
        st.dataframe(policies, use_container_width=True, hide_index=True)
    else:
        st.info("No scored policy rows are available yet.")


def _action_rows(review: dict[str, Any]) -> list[dict[str, str]]:
    quick = review.get("quick_iteration") or {}
    train = review.get("train") or {}
    render = review.get("render") or {}
    sections = ", ".join(review.get("changed_sections") or []) or "n/a"
    rows = [
        {
            "action": "Diagnose",
            "task": str((review.get("rationale") or {}).get("primary_failure") or "review rollout evidence"),
            "detail": _compact_list((review.get("rationale") or {}).get("secondary_failures") or []),
        },
        {
            "action": "Patch",
            "task": sections,
            "detail": f"{len(review.get('change_rows') or [])} parameter changes",
        },
    ]
    if train or quick:
        rows.append(
            {
                "action": "Train",
                "task": str(review.get("task") or "policy run"),
                "detail": _compact_json(
                    {
                        "seed": train.get("seed") or quick.get("seed"),
                        "num_envs": train.get("num_envs") or quick.get("num_envs"),
                        "max_iterations": train.get("max_iterations") or quick.get("max_iterations"),
                    }
                ),
            }
        )
    if render:
        rows.append(
            {
                "action": "Render",
                "task": str(render.get("mode") or "rollout"),
                "detail": _compact_json(
                    {
                        "video_length": render.get("video_length") or quick.get("video_length"),
                        "fps": render.get("fps"),
                        "num_envs": render.get("num_envs"),
                    }
                ),
            }
        )
    scenarios = review.get("generated_scenarios") or []
    if scenarios:
        rows.append(
            {
                "action": "Generate Scenarios",
                "task": f"{len(scenarios)} scenario tasks",
                "detail": ", ".join(str(item.get("scenario_id")) for item in scenarios[:4]),
            }
        )
    rows.append(
        {
            "action": "Review",
            "task": str(review.get("status")),
            "detail": f"score delta {format_score_delta(review.get('score_delta'))}",
        }
    )
    return rows


def _scenario_rows(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        rows.append(
            {
                "scenario_id": scenario.get("scenario_id"),
                "difficulty": scenario.get("difficulty"),
                "status": scenario.get("status"),
                "parent": scenario.get("parent_scenario_id"),
                "terrain": _compact_json(scenario.get("terrain") or {}),
                "disturbances": _compact_json(scenario.get("disturbances") or {}),
                "evaluation": _compact_json(scenario.get("evaluation") or {}),
            }
        )
    return rows


def _system_graph_dot(review: dict[str, Any]) -> str:
    rationale = review.get("rationale") or {}
    quick = review.get("quick_iteration") or {}
    train = review.get("train") or {}
    run_videos = [video for video in review.get("run_videos") or [] if video.get("exists")]
    source_videos = [video for video in review.get("source_videos") or [] if video.get("exists")]
    sections = review.get("changed_sections") or []
    scenario_count = len(review.get("generated_scenarios") or [])
    change_count = len(review.get("change_rows") or [])

    run_label = _short_text(str(review.get("experiment_id") or "selected run"), 36)
    source_label = f"Source renders\n{len(source_videos)} views" if source_videos else "Source renders\nnot loaded"
    failure_label = _short_text(str(rationale.get("primary_failure") or "review evidence"), 32)
    section_label = ", ".join(sections[:3]) if sections else "recorded patch"
    if len(sections) > 3:
        section_label = f"{section_label}, ..."
    render_label = f"Camera render\n{len(run_videos)} videos" if run_videos else "Camera render\nwaiting"
    train_detail = _compact_json(
        {
            "seed": train.get("seed") or quick.get("seed"),
            "envs": train.get("num_envs") or quick.get("num_envs"),
            "iters": train.get("max_iterations") or quick.get("max_iterations"),
        }
    )

    labels = {
        "source": source_label,
        "metrics": "Rollout evidence\nmetrics + video",
        "diagnose": f"Diagnose\n{failure_label}",
        "planner": "LLM planner\nhypothesis + risk",
        "patch": f"Patch\n{change_count} changes",
        "spec": f"Experiment spec\n{_short_text(train_detail, 30)}",
        "train": f"Train policy\n{_short_text(str(review.get('task') or 'task'), 32)}",
        "render": render_label,
        "review": f"Review\n{str(review.get('status') or 'n/a')}",
        "memory": "Research memory\nSQLite + artifacts",
        "scenarios": f"Scenario frontier\n{scenario_count} tasks",
        "run": f"Selected run\n{run_label}",
        "sections": f"Changed sections\n{_short_text(section_label, 34)}",
    }

    node_lines = "\n".join(
        f'    {node} [label="{_dot_escape(label)}"];'
        for node, label in labels.items()
    )
    return f"""
digraph autoresearch {{
  graph [
    rankdir=LR,
    bgcolor="transparent",
    pad=0.2,
    nodesep=0.42,
    ranksep=0.58,
    splines=ortho
  ];
  node [
    shape=rect,
    style="rounded,filled",
    fontname="Helvetica",
    fontsize=11,
    margin="0.12,0.08",
    color="#cbd5e1",
    fillcolor="#f8fafc",
    fontcolor="#0f172a"
  ];
  edge [
    color="#64748b",
    arrowsize=0.7,
    penwidth=1.5,
    fontname="Helvetica",
    fontsize=10,
    fontcolor="#475569"
  ];

  subgraph cluster_observe {{
    label="Observe";
    color="#dbeafe";
    style="rounded";
    source; metrics; diagnose;
  }}

  subgraph cluster_reason {{
    label="Reason";
    color="#e0e7ff";
    style="rounded";
    planner; patch; sections;
  }}

  subgraph cluster_execute {{
    label="Execute";
    color="#dcfce7";
    style="rounded";
    spec; train; render;
  }}

  subgraph cluster_review {{
    label="Review And Learn";
    color="#fee2e2";
    style="rounded";
    review; memory; scenarios; run;
  }}

{node_lines}

  run -> source [style=dashed, label="context"];
  source -> metrics;
  metrics -> diagnose;
  diagnose -> planner;
  planner -> patch;
  patch -> sections [label="bounded"];
  sections -> spec;
  spec -> train;
  train -> render;
  render -> review;
  review -> memory;
  memory -> planner [label="history"];
  planner -> scenarios [label="generate"];
  scenarios -> planner [label="frontier"];
  review -> run [style=dashed, label="current"];
}}
""".strip()


def _run_label(review: dict[str, Any]) -> str:
    return str(review["experiment_id"])


def _default_review_index(reviews: list[dict[str, Any]]) -> int:
    for index, review in enumerate(reviews):
        if any(video.get("exists") for video in review.get("run_videos") or []):
            return index
    return 0


def _policy_reference_count(policies: list[dict[str, Any]], reviews: list[dict[str, Any]]) -> int:
    policy_ids = {str(policy["policy_id"]) for policy in policies if policy.get("policy_id")}
    policy_ids.update(str(review["parent_policy_id"]) for review in reviews if review.get("parent_policy_id"))
    return len(policy_ids)


def _video_tab_label(video: dict[str, Any], index: int) -> str:
    label = str(video.get("label") or f"Video {index}")
    return label if len(label) <= 28 else f"{label[:25]}..."


def _compact_list(values: list[Any]) -> str:
    if not values:
        return "n/a"
    return ", ".join(str(value) for value in values[:5])


def _compact_json(value: dict[str, Any]) -> str:
    cleaned = {key: item for key, item in value.items() if item not in (None, "", [], {})}
    return json.dumps(cleaned, sort_keys=True) if cleaned else "n/a"


def _short_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: max(0, limit - 3)]}..."


def _dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _install_styles(st: Any) -> None:
    st.markdown(
        """
        <style>
          .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
          }
          h1, h2, h3, h4 {
            letter-spacing: 0;
          }
          .run-title {
            align-items: center;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            display: flex;
            gap: 12px;
            justify-content: space-between;
            margin: 0.25rem 0 1rem;
            padding: 12px 14px;
          }
          .run-title span:first-child {
            color: #111827;
            font-size: 0.95rem;
            font-weight: 700;
            overflow-wrap: anywhere;
          }
          .status-pill {
            background: #eef2ff;
            border: 1px solid #c7d2fe;
            border-radius: 999px;
            color: #3730a3;
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 700;
            line-height: 1;
            padding: 6px 9px;
            text-transform: uppercase;
            white-space: nowrap;
          }
          .status-pill.muted {
            background: #f8fafc;
            border-color: #e2e8f0;
            color: #334155;
            margin-bottom: 0.75rem;
          }
          .detail-grid {
            display: grid;
            gap: 8px;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            margin: 0.75rem 0 1.25rem;
          }
          .detail-cell {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            min-width: 0;
            padding: 10px 12px;
          }
          .detail-cell span {
            color: #64748b;
            display: block;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 4px;
            text-transform: uppercase;
          }
          .detail-cell strong {
            color: #111827;
            display: block;
            font-size: 0.88rem;
            overflow-wrap: anywhere;
          }
          [data-testid="stMetric"] {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 10px 12px;
          }
          div[data-testid="stVideo"] video {
            width: 100%;
          }
          @media (max-width: 900px) {
            .detail-grid {
              grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .run-title {
              align-items: flex-start;
              flex-direction: column;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
