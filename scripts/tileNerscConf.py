#!/usr/bin/env python3
"""Tile a NERSC gauge configuration periodically, e.g. 8^3x4 -> 16^3x4.

new(x,y,z,t) = old(x % Lx, y % Ly, z % Lz, t % Lt). The tiled field is again a valid
periodic configuration with the same plaquette and link trace, so it can be split over
more MPI ranks than the original (each local extent must exceed the halo depth).

Usage: tileNerscConf.py INPUT OUTPUT NX NY NZ NT   (tiles per direction, e.g. 2 2 2 1)
"""
import sys


def main():
    if len(sys.argv) != 7:
        sys.exit(__doc__)
    source, target = sys.argv[1], sys.argv[2]
    tiles = [int(n) for n in sys.argv[3:7]]
    if min(tiles) < 1:
        sys.exit("Tile counts must be positive")
    with open(source, "rb") as f:
        raw = f.read()
    end = raw.index(b"END_HEADER")
    body = raw.index(b"\n", end) + 1
    header = {}
    for line in raw[:end].decode().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            header[key.strip()] = value.strip()
    dims = [int(header[f"DIMENSION_{i}"]) for i in range(1, 5)]
    data = raw[body:]
    nsites = dims[0] * dims[1] * dims[2] * dims[3]
    if len(data) % nsites:
        sys.exit(f"Data size {len(data)} is not a multiple of {nsites} sites")
    site = len(data) // nsites  # bytes of the 4 links of one site
    lx, ly, lz, lt = dims
    # x runs fastest in the file: copy whole x-rows, repeated NX times.
    rows = []
    for t in range(lt * tiles[3]):
        for z in range(lz * tiles[2]):
            for y in range(ly * tiles[1]):
                start = (((t % lt) * lz + z % lz) * ly + y % ly) * lx * site
                rows.append(data[start:start + lx * site] * tiles[0])
    tiled = b"".join(rows)
    # NERSC checksum: sum of all 32-bit data words mod 2^32. Every word now occurs
    # prod(tiles) times.
    copies = tiles[0] * tiles[1] * tiles[2] * tiles[3]
    checksum = (int(header["CHECKSUM"], 16) * copies) % 2**32
    out = []
    for line in raw[:body].decode().splitlines():
        key = line.split("=", 1)[0].strip()
        if key.startswith("DIMENSION_"):
            i = int(key[-1]) - 1
            line = f"DIMENSION_{i + 1} = {dims[i] * tiles[i]}"
        elif key == "CHECKSUM":
            line = f"CHECKSUM = {checksum:x}"
        out.append(line)
    with open(target, "wb") as f:
        f.write(("\n".join(out) + "\n").encode())
        f.write(tiled)
    print(f"{source} {dims} -> {target} {[d * n for d, n in zip(dims, tiles)]}, CHECKSUM {checksum:x}")


if __name__ == "__main__":
    main()
