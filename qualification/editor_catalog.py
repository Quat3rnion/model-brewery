"""Generate the UI catalog from two successful, same-source native plan exports."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from profile_preview import aligned, compatible, digest, load_plan, read_json, require, role_of


def catalog(first, second):
    require(all(j["state"] == "succeeded" and j["kind"] == "plan" for j in (first, second)),
            "need successful plan jobs")
    a, b = first["result"], second["result"]
    require(re.fullmatch("[0-9a-f]{64}", a["source_sha256"]) and a["source_sha256"] == b["source_sha256"],
            "plans do not attest the same source")
    v, meta, assets, opts = load_plan(a["report"], "v2")
    h, meta2, assets2, opts2 = load_plan(b["report"], "ht43")
    # The editor compares file layout, not encoder throughput or numerical
    # results. The HT43 receipt may select a different unused I4R backend.
    require((meta, assets) == (meta2, assets2) and
            {k:v for k,v in opts.items() if k != "i4r_backend"} ==
            {k:v for k,v in opts2.items() if k != "i4r_backend"}, "incompatible plan options")
    edits, companions = compatible(v, h)
    rows = []
    for file, old in v.items():
        new = h[file]
        last = next(reversed(old.tensors))

        def difference(name):
            size = int if name == last else aligned
            return size(new.tensors[name].payload_bytes) - size(old.tensors[name].payload_bytes)

        for name, tensor in old.tensors.items():
            key = (file, name)
            layer, role = role_of(file, name)
            group = "global"
            if file.endswith("-vision.hgn"):
                group = "vision"
                m = re.match(r"visual\.blocks\.(\d+)\.", name)
                layer = int(m[1]) if m else None
            elif file.endswith("-ngram.hgn"):
                group = "engram"
            elif name.startswith("mtp."):
                group = "mtp"
                m = re.match(r"mtp\.layers\.(\d+)\.", name)
                layer = int(m[1]) if m else None
            elif layer is not None:
                group = "trunk"
            rows.append(dict(name=name, file=file, group=group, layer=layer,
                             role=role or ("rotation_companion" if tensor.origin.keys() & {"Signs","HtScales"} else "fixed"),
                             v2=tensor.encoding, ht43=new.tensors[name].encoding,
                             editable=key in edits, delta_bytes=(
                                 difference(name) + sum(difference(c) for c in companions.get(key, ()))
                                 if key in edits else 0)))
    sizes = {"v2": sum(c.file_size for c in v.values()), "ht43": sum(c.file_size for c in h.values())}
    require(sum(r["delta_bytes"] for r in rows) == sizes["ht43"] - sizes["v2"], "layout deltas do not reconcile")
    require(max(Counter(r["name"] for r in rows if r["editable"]).values()) == 1, "ambiguous editable name")
    data = dict(schema=1, architecture="qwen4_exp", source_sha256=a["source_sha256"], base_bytes=sizes, tensors=rows)
    data["catalog_sha256"] = digest(data)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("v2", type=Path)
    parser.add_argument("ht43", type=Path)
    args = parser.parse_args()
    print(json.dumps(catalog(read_json(args.v2, 8 << 20), read_json(args.ht43, 8 << 20)), separators=(",", ":")))
