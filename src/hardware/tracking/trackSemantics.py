from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SemanticEvent:
    semantic_id: str
    semantic_type: str
    label: str
    anchor_node_id: str | None = None
    control_type: str | None = None
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = {
            "id": self.semantic_id,
            "type": self.semantic_type,
            "label": self.label,
            "anchor_node_id": self.anchor_node_id,
        }
        if self.control_type:
            payload["control_type"] = self.control_type
        if self.data:
            payload.update(self.data)
        return payload


class TrackSemantics:
    """Loads optional route semantics stored alongside the GraphML track."""

    def __init__(self, path: str | None = None):
        self.path = path
        self.loaded = False
        self.version = 1
        self.map_metadata: dict = {}
        self.destinations: list[dict] = []
        self.controls: list[dict] = []
        self.intersections: list[dict] = []
        self.crosswalks: list[dict] = []
        self.parking_spots: list[dict] = []
        self.zones: list[dict] = []
        self.route_annotation_defaults: dict = {}

        self._destinations_by_id: dict[str, dict] = {}
        self._destinations_by_label: dict[str, dict] = {}
        self._controls_by_id: dict[str, dict] = {}
        self._node_events: dict[str, list[SemanticEvent]] = defaultdict(list)

        if path:
            self._load(path)

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        self.loaded = True
        self.version = int(data.get("version", 1) or 1)
        self.map_metadata = dict(data.get("map_metadata") or {})
        self.route_annotation_defaults = dict(data.get("route_annotations_defaults") or {})

        self.destinations = self._normalize_records(data.get("destinations"))
        self.controls = self._normalize_records(data.get("controls"))
        self.intersections = self._normalize_records(data.get("intersections"))
        self.crosswalks = self._normalize_records(data.get("crosswalks"))
        self.parking_spots = self._normalize_records(data.get("parking_spots"))
        self.zones = self._normalize_records(data.get("zones"))

        for destination in self.destinations:
            dest_id = str(destination.get("id") or "").strip()
            node_id = self._norm_node_id(destination.get("node_id"))
            if not dest_id or not node_id:
                continue
            destination["id"] = dest_id
            destination["node_id"] = node_id
            destination.setdefault("label", dest_id)
            self._destinations_by_id[dest_id] = destination
            self._destinations_by_label[str(destination.get("label", "")).strip().lower()] = destination
            self._add_node_event(
                node_id,
                SemanticEvent(
                    semantic_id=dest_id,
                    semantic_type="destination",
                    label=str(destination.get("label") or dest_id),
                    anchor_node_id=node_id,
                    data={"destination_id": dest_id},
                ),
            )

        for control in self.controls:
            control_id = str(control.get("id") or "").strip()
            node_id = self._norm_node_id(control.get("anchor_node_id"))
            if not control_id or not node_id:
                continue
            control_type = str(control.get("type") or "generic").strip().lower()
            control["id"] = control_id
            control["anchor_node_id"] = node_id
            control["type"] = control_type
            control.setdefault("label", control_id)
            self._controls_by_id[control_id] = control
            self._add_node_event(
                node_id,
                SemanticEvent(
                    semantic_id=control_id,
                    semantic_type="control",
                    label=str(control.get("label") or control_id),
                    anchor_node_id=node_id,
                    control_type=control_type,
                    data={"control_id": control_id},
                ),
            )

        for intersection in self.intersections:
            inter_id = str(intersection.get("id") or "").strip()
            node_id = self._norm_node_id(intersection.get("anchor_node_id"))
            if not inter_id or not node_id:
                continue
            control_id = str(intersection.get("control_id") or "").strip() or None
            control = self._controls_by_id.get(control_id or "")
            intersection["id"] = inter_id
            intersection["anchor_node_id"] = node_id
            if control_id:
                intersection["control_id"] = control_id
            intersection.setdefault("label", inter_id)
            self._add_node_event(
                node_id,
                SemanticEvent(
                    semantic_id=inter_id,
                    semantic_type="intersection",
                    label=str(intersection.get("label") or inter_id),
                    anchor_node_id=node_id,
                    control_type=(control or {}).get("type"),
                    data={
                        "intersection_id": inter_id,
                        "control_id": control_id,
                        "stopline_node_id": self._norm_node_id(intersection.get("stopline_node_id")),
                    },
                ),
            )

        for crosswalk in self.crosswalks:
            crosswalk_id = str(crosswalk.get("id") or "").strip()
            node_id = self._norm_node_id(crosswalk.get("anchor_node_id") or crosswalk.get("stopline_node_id"))
            if not crosswalk_id or not node_id:
                continue
            crosswalk["id"] = crosswalk_id
            crosswalk.setdefault("anchor_node_id", node_id)
            crosswalk["anchor_node_id"] = self._norm_node_id(crosswalk.get("anchor_node_id") or node_id)
            crosswalk.setdefault("label", crosswalk_id)
            self._add_node_event(
                node_id,
                SemanticEvent(
                    semantic_id=crosswalk_id,
                    semantic_type="crosswalk",
                    label=str(crosswalk.get("label") or crosswalk_id),
                    anchor_node_id=node_id,
                    data={"crosswalk_id": crosswalk_id},
                ),
            )

        for parking in self.parking_spots:
            parking_id = str(parking.get("id") or "").strip()
            node_id = self._norm_node_id(parking.get("anchor_node_id") or parking.get("entry_node_id"))
            if not parking_id or not node_id:
                continue
            parking["id"] = parking_id
            parking.setdefault("anchor_node_id", node_id)
            parking["anchor_node_id"] = self._norm_node_id(parking.get("anchor_node_id") or node_id)
            parking.setdefault("entry_node_id", self._norm_node_id(parking.get("entry_node_id") or node_id))
            parking.setdefault("label", parking_id)
            self._add_node_event(
                node_id,
                SemanticEvent(
                    semantic_id=parking_id,
                    semantic_type="parking_spot",
                    label=str(parking.get("label") or parking_id),
                    anchor_node_id=node_id,
                    data={
                        "parking_spot_id": parking_id,
                        "entry_node_id": self._norm_node_id(parking.get("entry_node_id") or node_id),
                        "exit_node_id": self._norm_node_id(parking.get("exit_node_id")),
                    },
                ),
            )

        for zone in self.zones:
            zone_id = str(zone.get("id") or "").strip()
            if not zone_id:
                continue
            zone["id"] = zone_id
            zone.setdefault("label", zone_id)
            zone["start_node_id"] = self._norm_node_id(zone.get("start_node_id"))
            zone["end_node_id"] = self._norm_node_id(zone.get("end_node_id"))
            zone["type"] = str(zone.get("type") or "zone").strip().lower()

    @staticmethod
    def _normalize_records(records) -> list[dict]:
        if isinstance(records, list):
            return [dict(item) for item in records if isinstance(item, dict)]
        return []

    @staticmethod
    def _norm_node_id(node_id) -> str | None:
        if node_id is None:
            return None
        text = str(node_id).strip()
        return text or None

    def _add_node_event(self, node_id: str, event: SemanticEvent) -> None:
        node_key = str(node_id)
        existing = {
            (item.semantic_id, item.semantic_type)
            for item in self._node_events.get(node_key, [])
        }
        if (event.semantic_id, event.semantic_type) in existing:
            return
        self._node_events[node_key].append(event)

    def resolve_destination(self, spec) -> dict | None:
        if spec is None:
            return None
        if isinstance(spec, dict):
            destination_id = str(spec.get("destination_id") or spec.get("id") or "").strip()
            if destination_id and destination_id in self._destinations_by_id:
                return self._destinations_by_id[destination_id]
            label = str(spec.get("label") or spec.get("destination") or "").strip().lower()
            if label and label in self._destinations_by_label:
                return self._destinations_by_label[label]
            return None

        key = str(spec).strip()
        if key in self._destinations_by_id:
            return self._destinations_by_id[key]
        return self._destinations_by_label.get(key.lower())

    def get_available_destinations(self) -> list[dict]:
        return [
            {
                "id": str(item.get("id")),
                "label": str(item.get("label") or item.get("id")),
                "node_id": str(item.get("node_id")),
            }
            for item in self.destinations
            if item.get("id") and item.get("node_id")
        ]

    def get_node_events(self, node_id: str | None) -> list[dict]:
        if node_id is None:
            return []
        return [event.to_dict() for event in self._node_events.get(str(node_id), [])]

    def get_zone_records(self) -> list[dict]:
        return [dict(item) for item in self.zones]

    def build_route_annotations(self, node_ids: list[str], closed_loop: bool = False) -> dict:
        clean_ids = [str(node_id) for node_id in node_ids]
        node_events_by_index: list[list[dict]] = []
        zone_ids_by_index: list[list[str]] = []
        zone_types_by_index: list[list[str]] = []
        zone_labels_by_index: list[list[str]] = []
        route_events: list[dict] = []
        seen_event_ids: set[tuple[str, str]] = set()

        for idx, node_id in enumerate(clean_ids):
            node_events = self.get_node_events(node_id)
            node_events_by_index.append(node_events)
            zone_ids_by_index.append([])
            zone_types_by_index.append([])
            zone_labels_by_index.append([])
            for event in node_events:
                key = (str(event.get("type")), str(event.get("id")))
                if key in seen_event_ids:
                    continue
                seen_event_ids.add(key)
                route_events.append({**event, "route_node_index": idx, "node_id": node_id})

        for zone in self.zones:
            zone_id = str(zone.get("id") or "").strip()
            start_id = self._norm_node_id(zone.get("start_node_id"))
            end_id = self._norm_node_id(zone.get("end_node_id"))
            if not zone_id or not start_id or not end_id:
                continue
            covered_indexes = self._route_indexes_between(
                clean_ids,
                start_id,
                end_id,
                closed_loop=bool(closed_loop),
            )
            if not covered_indexes:
                continue
            zone_type = str(zone.get("type") or "zone")
            zone_label = str(zone.get("label") or zone_id)
            for idx in covered_indexes:
                zone_ids_by_index[idx].append(zone_id)
                zone_types_by_index[idx].append(zone_type)
                zone_labels_by_index[idx].append(zone_label)
            key = ("zone", zone_id)
            if key not in seen_event_ids:
                seen_event_ids.add(key)
                route_events.append(
                    {
                        "id": zone_id,
                        "type": "zone",
                        "label": zone_label,
                        "zone_type": zone_type,
                        "route_node_index": covered_indexes[0],
                        "node_id": clean_ids[covered_indexes[0]],
                    }
                )

        route_events.sort(key=lambda item: (int(item.get("route_node_index", 0)), str(item.get("id", ""))))
        return {
            "node_events_by_index": node_events_by_index,
            "zone_ids_by_index": zone_ids_by_index,
            "zone_types_by_index": zone_types_by_index,
            "zone_labels_by_index": zone_labels_by_index,
            "route_events": route_events,
        }

    @staticmethod
    def _route_indexes_between(
        node_ids: list[str],
        start_node_id: str,
        end_node_id: str,
        closed_loop: bool = False,
    ) -> list[int]:
        if not node_ids or start_node_id not in node_ids or end_node_id not in node_ids:
            return []
        start_idx = node_ids.index(start_node_id)
        end_idx = node_ids.index(end_node_id)
        if start_idx <= end_idx:
            return list(range(start_idx, end_idx + 1))
        if not closed_loop:
            return []
        return list(range(start_idx, len(node_ids))) + list(range(0, end_idx + 1))

