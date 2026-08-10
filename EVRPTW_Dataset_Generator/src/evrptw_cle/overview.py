from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from .visualization import component_color


def _short_city(label: str) -> str:
    return label.split(",")[0]


def _map_thumbnail(path: Path, max_width: int = 760) -> Image.Image:
    image = Image.open(path).convert("RGB")
    # Per-city PNGs reserve their top band for a title and right side for a
    # printed legend. The overview supplies both, so retain the map region only.
    image = image.crop((0, round(image.height * 0.07), round(image.width * 0.72), image.height))
    if image.width > max_width:
        height = round(image.height * max_width / image.width)
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    return image


def _encode_webp(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=72, method=6)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_overview(
    preset_file: Path,
    city_root: Path,
    output_dir: Path,
    *,
    fragment_file: Path | None = None,
) -> dict:
    preset = json.loads(preset_file.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    panels = []
    thumbnails: list[tuple[str, Image.Image, dict]] = []
    operational_thumbnails: list[tuple[str, Image.Image, dict]] = []

    for item in preset["cities"]:
        city_dir = city_root / item["slug"]
        manifest = json.loads((city_dir / "manifest.json").read_text(encoding="utf-8"))
        connectivity = manifest["connectivity"]
        components = pd.read_csv(city_dir / "components.csv")
        label = _short_city(manifest["city_label"])
        thumbnail = _map_thumbnail(city_dir / manifest["visualization"]["png"])
        encoded = _encode_webp(thumbnail)
        operational = manifest.get("operational_connectivity") or {}
        operational_visual = manifest.get("operational_visualization") or {}
        operational_thumbnail = _map_thumbnail(city_dir / operational_visual["png"])
        operational_encoded = _encode_webp(operational_thumbnail)
        largest_share = float(connectivity["largest_weak_component_node_share"])
        record = {
            "slug": item["slug"],
            "city": label,
            "node_count": int(connectivity["node_count"]),
            "directed_edge_count": int(connectivity["directed_edge_count"]),
            "weak_component_count": int(connectivity["weak_component_count"]),
            "largest_weak_component_nodes": int(connectivity["largest_weak_component_nodes"]),
            "largest_weak_component_node_share": largest_share,
            "non_largest_component_nodes": int(connectivity["node_count"])
            - int(connectivity["largest_weak_component_nodes"]),
            "component_policy": manifest["component_policy"],
            "selected_graph": manifest["selected_graph"],
            "pbf_replication_timestamp_utc": manifest.get("provenance", {})
            .get("osm", {})
            .get("pbf_replication_timestamp_utc"),
            "operational_selected_buffer_km": float(operational["selected_buffer_km"]),
            "operational_city_node_coverage": float(operational["city_node_coverage"]),
            "operational_city_road_length_coverage": float(
                operational["city_physical_road_length_coverage"]
            ),
            "operational_node_count": int(operational["operational_node_count"]),
            "operational_transit_only_node_count": int(operational["transit_only_node_count"]),
        }
        records.append(record)
        thumbnails.append((label, thumbnail, record))
        operational_thumbnails.append((label, operational_thumbnail, record))

        legend = []
        for row in components.itertuples(index=False):
            legend.append(
                "<li>"
                f'<span class="swatch" style="--c:{component_color(int(row.rank))}"></span>'
                f"<code>{html.escape(str(row.component_id))}</code>"
                f"<span>{int(row.node_count):,} nodes · "
                f"{float(row.node_share) * 100:.4f}%</span>"
                "</li>"
            )
        panels.append(
            f"""<article class="city-card">
<div class="card-head"><div><h2>{html.escape(label)}</h2>
<p>{record["node_count"]:,} nodes · {record["directed_edge_count"]:,} directed edges</p></div>
<span class="wcc">{record["weak_component_count"]} WCC</span></div>
<div class="maps"><figure><figcaption>Exact city graph</figcaption><img src="data:image/webp;base64,{encoded}" alt="{html.escape(label)} road graph colored by weak component"></figure>
<figure><figcaption>Operational graph</figcaption><img src="data:image/webp;base64,{operational_encoded}" alt="{html.escape(label)} operational routing graph"></figure></div>
<div class="metric"><span>Main component</span><strong>{largest_share * 100:.3f}%</strong></div>
<div class="bar"><i style="width:{largest_share * 100:.5f}%"></i></div>
<p class="small">W0001 contains {record["largest_weak_component_nodes"]:,} nodes; "
{record["non_largest_component_nodes"]:,} nodes remain in smaller components. Operational selection uses a {record["operational_selected_buffer_km"]:g} km buffer and covers {record["operational_city_node_coverage"] * 100:.3f}% of city nodes and {record["operational_city_road_length_coverage"] * 100:.3f}% of physical road length.</p>
<details><summary>Exact component legend ({record["weak_component_count"]})</summary>
<ul class="legend">{"".join(legend)}</ul></details>
</article>"""
        )

    summary = pd.DataFrame.from_records(records)
    summary_path = output_dir / "top10_connectivity_summary.csv"
    summary.to_csv(summary_path, index=False)

    figure, axes = plt.subplots(5, 2, figsize=(16, 24), dpi=120)
    figure.patch.set_facecolor("#f5f7f8")
    for axis, (label, image, record) in zip(axes.ravel(), thumbnails, strict=True):
        axis.imshow(image)
        axis.set_title(
            f"{label}  ·  {record['weak_component_count']} WCC  ·  "
            f"W0001 {record['largest_weak_component_node_share'] * 100:.3f}%",
            loc="left",
            fontsize=11,
        )
        axis.axis("off")
    figure.suptitle(
        "Ten U.S. city drive graphs — all weak components retained",
        fontsize=18,
        x=0.03,
        ha="left",
    )
    figure.text(
        0.03,
        0.008,
        "Teal = W0001 (largest weak component). Other colors = smaller weak components; "
        "see per-city HTML/CSV for the exact legend.",
        fontsize=10,
        color="#4b5b65",
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    png_path = output_dir / "top10_connectivity_overview.png"
    figure.savefig(png_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)

    operational_figure, operational_axes = plt.subplots(5, 2, figsize=(16, 24), dpi=120)
    operational_figure.patch.set_facecolor("#f5f7f8")
    for axis, (label, image, record) in zip(
        operational_axes.ravel(), operational_thumbnails, strict=True
    ):
        axis.imshow(image)
        axis.set_title(
            f"{label} · buffer {record['operational_selected_buffer_km']:g} km · "
            f"nodes {record['operational_city_node_coverage'] * 100:.3f}% · "
            f"length {record['operational_city_road_length_coverage'] * 100:.3f}%",
            loc="left",
            fontsize=10,
        )
        axis.axis("off")
    operational_figure.suptitle(
        "Ten U.S. city operational routing graphs — actual OSM connectors only",
        fontsize=18,
        x=0.03,
        ha="left",
    )
    operational_figure.text(
        0.03,
        0.008,
        "Teal = inside-city service roads. Orange = outside-city transit-only OSM roads. No synthetic connector edges.",
        fontsize=10,
        color="#4b5b65",
    )
    operational_figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    operational_png_path = output_dir / "top10_operational_overview.png"
    operational_figure.savefig(
        operational_png_path,
        bbox_inches="tight",
        pad_inches=0.12,
    )
    plt.close(operational_figure)

    style = """
<style>
.evr-overview{--bg:#f4f7f8;--card:#fff;--fg:#12212a;--muted:#5e6d76;--rule:#d8e0e4;--main:#006d8f;background:var(--bg);color:var(--fg);padding:18px;border-radius:18px;font-family:ui-sans-serif,system-ui,sans-serif}.evr-overview *{box-sizing:border-box}.evr-overview header{max-width:980px;margin:0 auto 18px}.evr-overview h1{font-size:clamp(24px,4vw,40px);line-height:1.05;margin:0 0 8px;font-weight:620}.evr-overview header p,.evr-overview .small,.evr-overview .card-head p{color:var(--muted)}.evr-overview .key{display:flex;gap:18px;flex-wrap:wrap;margin-top:12px;font-size:13px}.evr-overview .key span{display:inline-flex;gap:7px;align-items:center}.evr-overview .dot{width:13px;height:5px;background:var(--main);display:inline-block}.evr-overview .dot.other{background:linear-gradient(90deg,#c25d3d,#6e8c3a,#7855a6)}.evr-overview .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;max-width:1320px;margin:auto}.evr-overview .city-card{background:var(--card);border:1px solid var(--rule);border-radius:14px;padding:14px;box-shadow:0 8px 24px rgba(16,37,48,.05)}.evr-overview .card-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}.evr-overview h2{font-size:21px;margin:0 0 2px}.evr-overview .card-head p{font-size:12px;margin:0}.evr-overview .wcc{background:#e5f1f4;color:#005a76;border-radius:999px;padding:5px 9px;font-size:12px;white-space:nowrap}.evr-overview .maps{display:grid;grid-template-columns:1fr 1fr;gap:8px}.evr-overview figure{margin:6px 0}.evr-overview figcaption{font-size:11px;color:var(--muted)}.evr-overview img{display:block;width:100%;height:auto;margin:4px 0 6px}.evr-overview .metric{display:flex;justify-content:space-between;font-size:13px}.evr-overview .bar{height:6px;background:#e3e8eb;border-radius:9px;overflow:hidden;margin:5px 0}.evr-overview .bar i{height:100%;display:block;background:var(--main)}.evr-overview .small{font-size:12px;margin:7px 0 10px}.evr-overview details{border-top:1px solid var(--rule);padding-top:8px}.evr-overview summary{cursor:pointer;font-size:13px}.evr-overview .legend{list-style:none;padding:8px 0 0;margin:0;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:5px 12px}.evr-overview .legend li{display:grid;grid-template-columns:15px 52px 1fr;align-items:center;gap:5px;font-size:11px;color:var(--muted)}.evr-overview .swatch{height:4px;background:var(--c)}.evr-overview code{color:var(--fg);font-size:11px}@media(max-width:780px){.evr-overview .grid,.evr-overview .maps{grid-template-columns:1fr}.evr-overview .legend{grid-template-columns:1fr}}@media(prefers-color-scheme:dark){.evr-overview{--bg:#11181d;--card:#172128;--fg:#edf3f5;--muted:#a2afb6;--rule:#34424a}.evr-overview .wcc{background:#173945;color:#9ad8e9}}
</style>
"""
    fragment = f"""{style}<section class="evr-overview">
<header><h1>Ten city raw and operational OSM road graphs</h1>
<p>Each city preserves the exact-boundary raw graph for audit and publishes one connected operational graph using actual OSM transit-only connector roads.</p>
<div class="key"><span><i class="dot"></i>W0001: largest weak component</span><span><i class="dot other"></i>W0002+: smaller components</span></div></header>
<div class="grid">{"".join(panels)}</div></section>"""
    html_path = output_dir / "top10_connectivity_overview.html"
    html_path.write_text(
        '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        + style
        + '</head><body style="margin:0">'
        + fragment.removeprefix(style)
        + "</body></html>",
        encoding="utf-8",
    )
    if fragment_file is not None:
        fragment_file.parent.mkdir(parents=True, exist_ok=True)
        fragment_file.write_text(fragment, encoding="utf-8")
    return {
        "city_count": len(records),
        "png": str(png_path),
        "operational_png": str(operational_png_path),
        "html": str(html_path),
        "summary_csv": str(summary_path),
        "fragment": str(fragment_file) if fragment_file is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--city-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fragment-file", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_overview(
                args.preset,
                args.city_root,
                args.output_dir,
                fragment_file=args.fragment_file,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
