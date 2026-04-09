#!/usr/bin/env python3
"""
Mide la distancia en metros entre nodos del grafo del circuito.

Uso:
    python tools/node_distance.py <nodo_a> <nodo_b> [nodo_c ...]
    python tools/node_distance.py --list
    python tools/node_distance.py --all <nodo_base>

Ejemplos:
    python tools/node_distance.py 46 96
    python tools/node_distance.py 46 59 94 96
    python tools/node_distance.py --all 46
"""

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

GRAPHML = os.path.join(os.path.dirname(__file__), "..", "Track GraphML File.graphml")
NS = "http://graphml.graphdrawing.org/graphml"


def load_nodes(path: str) -> dict[str, tuple[float, float, int, bool]]:
    tree = ET.parse(path)
    root = tree.getroot()

    key_map = {}
    for key in root.findall(f"{{{NS}}}key"):
        key_map[key.get("attr.name", "")] = key.get("id", "")

    x_key = key_map.get("x", "d0")
    y_key = key_map.get("y", "d1")
    attr_key = key_map.get("new_attribute", "d2")
    start_key = key_map.get("is_start", "d3")

    graph = root.find(f"{{{NS}}}graph")
    nodes = {}
    for node_el in graph.findall(f"{{{NS}}}node"):
        nid = node_el.get("id")
        if nid is None:
            continue
        data = {d.get("key"): d.text for d in node_el.findall(f"{{{NS}}}data")}
        x = float(data.get(x_key, 0.0))
        y = float(data.get(y_key, 0.0))
        attr = int(data.get(attr_key, 0) or 0)
        is_start = str(data.get(start_key, "false")).lower() == "true"
        nodes[nid] = (x, y, attr, is_start)
    return nodes


ATTR_NAMES = {
    0: "normal", 1: "crosswalk", 2: "intersection", 3: "oneway",
    4: "highway_left", 5: "highway_right", 6: "roundabout",
    7: "stopline", 8: "dotted", 9: "dotted_crosswalk",
}


def dist(a: tuple, b: tuple) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def fmt_node(nid: str, data: tuple) -> str:
    x, y, attr, is_start = data
    tag = " [START]" if is_start else ""
    return f"  {nid:>4}  ({x:.4f}, {y:.4f})  attr={ATTR_NAMES.get(attr, attr)}{tag}"


def main():
    parser = argparse.ArgumentParser(description="Distancia entre nodos del circuito")
    parser.add_argument("nodes", nargs="*", help="IDs de nodos (mínimo 2)")
    parser.add_argument("--list", action="store_true", help="Listar todos los nodos")
    parser.add_argument("--all", metavar="NODO", help="Distancia desde NODO a todos los demás")
    parser.add_argument("--graphml", default=GRAPHML, help="Ruta al archivo GraphML")
    args = parser.parse_args()

    graphml_path = os.path.abspath(args.graphml)
    if not os.path.exists(graphml_path):
        print(f"ERROR: No se encontró {graphml_path}", file=sys.stderr)
        sys.exit(1)

    nodes = load_nodes(graphml_path)
    sorted_ids = sorted(nodes.keys(), key=lambda n: int(n))

    if args.list:
        print(f"{'ID':>4}  {'X (m)':>10}  {'Y (m)':>10}  Tipo")
        print("-" * 50)
        for nid in sorted_ids:
            x, y, attr, is_start = nodes[nid]
            tag = " [START]" if is_start else ""
            print(f"{nid:>4}  {x:>10.4f}  {y:>10.4f}  {ATTR_NAMES.get(attr, attr)}{tag}")
        print(f"\nTotal: {len(nodes)} nodos")
        return

    if args.all:
        base_id = args.all
        if base_id not in nodes:
            print(f"ERROR: Nodo '{base_id}' no existe", file=sys.stderr)
            sys.exit(1)
        base = nodes[base_id]
        print(f"Distancias desde nodo {base_id}:")
        print(fmt_node(base_id, base))
        print()
        results = [(dist(base, nodes[nid]), nid) for nid in sorted_ids if nid != base_id]
        results.sort()
        print(f"{'Nodo':>4}  {'Distancia':>10}  Tipo")
        print("-" * 40)
        for d, nid in results:
            x, y, attr, _ = nodes[nid]
            print(f"{nid:>4}  {d:>8.4f} m  {ATTR_NAMES.get(attr, attr)}")
        return

    node_ids = args.nodes
    if len(node_ids) < 2:
        parser.print_help()
        sys.exit(1)

    missing = [n for n in node_ids if n not in nodes]
    if missing:
        print(f"ERROR: Nodos no encontrados: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    print("Nodos:")
    for nid in node_ids:
        print(fmt_node(nid, nodes[nid]))
    print()

    if len(node_ids) == 2:
        a, b = node_ids
        d = dist(nodes[a], nodes[b])
        print(f"Distancia {a} → {b}: {d:.4f} m  ({d*100:.1f} cm)")
    else:
        total = 0.0
        print(f"{'Tramo':>12}  {'Distancia':>10}")
        print("-" * 30)
        for i in range(len(node_ids) - 1):
            a, b = node_ids[i], node_ids[i + 1]
            d = dist(nodes[a], nodes[b])
            total += d
            print(f"{a:>4} → {b:>4}   {d:>8.4f} m")
        print("-" * 30)
        print(f"{'Total':>12}  {total:>8.4f} m  ({total*100:.1f} cm)")


if __name__ == "__main__":
    main()
