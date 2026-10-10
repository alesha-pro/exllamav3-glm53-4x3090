#!/usr/bin/env python3
"""Write a mixed EXL3 quant as one model directory.

The base quant is copied shard by shard; every tensor whose key matches --re is replaced by the tensor of the same
key from the second quant (all *.safetensors of that directory are searched, like EXL3_MIX_DIR does at load time).
With --verify every tensor of the result is compared byte for byte with its source.

  merge_mix.py --base <2.05 bpw dir> --mix <3.05 bpw dir> --out <new dir> [--verify]
"""
import argparse, glob, json, os, re, shutil, struct, time

CHUNK = 64 << 20


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def copy_range(src, begin, end, dst):
    src.seek(begin)
    left = end - begin
    while left:
        buf = src.read(min(CHUNK, left))
        assert buf, "short read"
        dst.write(buf)
        left -= len(buf)


def same_range(fa, a0, fb, b0, size):
    fa.seek(a0); fb.seek(b0)
    while size:
        n = min(CHUNK, size)
        if fa.read(n) != fb.read(n):
            return False
        size -= n
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--mix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--re", default=r"\.mlp\.experts\.\d+\.down_proj\.")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-only", action="store_true", help="compare an existing --out with its sources")
    a = ap.parse_args()
    rx = re.compile(a.re)
    if not a.verify_only:
        os.makedirs(a.out, exist_ok=False)

    # key -> (file, data start, entry) for the matching tensors of the second quant
    donor = {}
    for path in sorted(glob.glob(os.path.join(a.mix, "*.safetensors"))):
        hdr, data0 = read_header(path)
        for k, e in hdr.items():
            if k != "__metadata__" and rx.search(k):
                assert k not in donor, f"{k} is in two files of {a.mix}"
                donor[k] = (path, data0, e)
    print(f"donor tensors matching the pattern: {len(donor)}", flush=True)

    plan, used, t0 = {}, set(), time.time()   # out file -> [(key, src path, abs begin, abs end)]
    base_bytes = out_bytes = 0
    for path in sorted(glob.glob(os.path.join(a.base, "*.safetensors"))):
        hdr, data0 = read_header(path)
        meta = hdr.pop("__metadata__", None)
        keys = sorted(hdr, key=lambda k: hdr[k]["data_offsets"][0])
        new_hdr, items, off = {}, [], 0
        for k in keys:
            e = hdr[k]
            base_bytes += e["data_offsets"][1] - e["data_offsets"][0]
            if rx.search(k):
                assert k in donor, f"{k} is missing in {a.mix}"
                src, s0, e = donor[k]
                used.add(k)
            else:
                src, s0 = path, data0
            b, n = e["data_offsets"][0], e["data_offsets"][1] - e["data_offsets"][0]
            new_hdr[k] = {"dtype": e["dtype"], "shape": e["shape"], "data_offsets": [off, off + n]}
            items.append((k, src, s0 + b, s0 + b + n))
            off += n
        out_bytes += off
        if meta is not None:
            new_hdr["__metadata__"] = meta
        raw = json.dumps(new_hdr, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        out_path = os.path.join(a.out, os.path.basename(path))
        handles = {}
        if a.verify_only:
            with open(out_path, "rb") as f:
                assert f.read(8 + len(raw)) == struct.pack("<Q", len(raw)) + raw, f"header of {out_path} differs"
        else:
            with open(out_path, "wb") as dst:
                dst.write(struct.pack("<Q", len(raw))); dst.write(raw)
                for k, src, b, e in items:
                    if src not in handles:
                        handles[src] = open(src, "rb")
                    copy_range(handles[src], b, e, dst)
        for h in handles.values():
            h.close()
        plan[out_path] = (8 + len(raw), new_hdr, items)
        print(f"{os.path.basename(path)}: {len(keys)} tensors, {off / 2**30:.2f} GiB, {time.time() - t0:.0f} s", flush=True)
    assert used == set(donor), f"{len(set(donor) - used)} donor tensors have no place in the base quant"

    for name in os.listdir(a.base):
        src = os.path.join(a.base, name)
        if os.path.isfile(src) and not name.endswith(".safetensors") and not a.verify_only:
            shutil.copy2(src, os.path.join(a.out, name))
    print(f"replaced {len(used)} tensors; tensor bytes {base_bytes / 2**30:.2f} -> {out_bytes / 2**30:.2f} GiB", flush=True)

    if a.verify or a.verify_only:
        bad = n = 0
        for out_path, (data0, new_hdr, items) in plan.items():
            handles = {}
            with open(out_path, "rb") as fo:
                for k, src, b, e in items:
                    if src not in handles:
                        handles[src] = open(src, "rb")
                    n += 1
                    if not same_range(fo, data0 + new_hdr[k]["data_offsets"][0], handles[src], b, e - b):
                        bad += 1
                        print("MISMATCH", k, flush=True)
            for h in handles.values():
                h.close()
        print(f"verify: {n} tensors compared with their sources, {bad} mismatches", flush=True)
    print(f"MERGE DONE {a.out} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
