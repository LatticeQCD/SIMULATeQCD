#!/usr/bin/env python3

"""Create review-ready PDF plots from the HISQ benchmark CSV files.

The PDF writer uses only the Python standard library. The CSV files remain the
quantitative source data, and no matplotlib installation is required.
"""

import argparse
import csv
import math
from pathlib import Path


WIDTH = 900
HEIGHT = 560
LEFT = 92
RIGHT = 32
TOP = 68
BOTTOM = 82
COLORS = ("#2458a6", "#d1495b", "#27864b")


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"No data rows in {path}")
    return rows


def pdf_text(value):
    value = str(value).encode("ascii", "replace").decode("ascii")
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def rgb(hex_color):
    return tuple(int(hex_color[index:index + 2], 16) / 255.0 for index in (1, 3, 5))


class PdfCanvas:
    def __init__(self):
        self.commands = []

    @staticmethod
    def y(value):
        return HEIGHT - value

    def line(self, x1, y1, x2, y2, color="#222222", width=1.0):
        red, green, blue = rgb(color)
        self.commands.append(
            f"{red:.4f} {green:.4f} {blue:.4f} RG {width:.2f} w "
            f"{x1:.2f} {self.y(y1):.2f} m {x2:.2f} {self.y(y2):.2f} l S")

    def polyline(self, points, color, width=3.0):
        red, green, blue = rgb(color)
        first_x, first_y = points[0]
        path = [f"{first_x:.2f} {self.y(first_y):.2f} m"]
        path.extend(f"{x:.2f} {self.y(y):.2f} l" for x, y in points[1:])
        self.commands.append(
            f"{red:.4f} {green:.4f} {blue:.4f} RG {width:.2f} w " + " ".join(path) + " S")

    def rectangle(self, x, y, width, height, color):
        red, green, blue = rgb(color)
        self.commands.append(
            f"{red:.4f} {green:.4f} {blue:.4f} rg "
            f"{x:.2f} {self.y(y + height):.2f} {width:.2f} {height:.2f} re f")

    def circle(self, center_x, center_y, radius, color):
        red, green, blue = rgb(color)
        center_y = self.y(center_y)
        control = radius * 0.5522847498
        self.commands.append(
            f"{red:.4f} {green:.4f} {blue:.4f} rg "
            f"{center_x + radius:.2f} {center_y:.2f} m "
            f"{center_x + radius:.2f} {center_y + control:.2f} "
            f"{center_x + control:.2f} {center_y + radius:.2f} "
            f"{center_x:.2f} {center_y + radius:.2f} c "
            f"{center_x - control:.2f} {center_y + radius:.2f} "
            f"{center_x - radius:.2f} {center_y + control:.2f} "
            f"{center_x - radius:.2f} {center_y:.2f} c "
            f"{center_x - radius:.2f} {center_y - control:.2f} "
            f"{center_x - control:.2f} {center_y - radius:.2f} "
            f"{center_x:.2f} {center_y - radius:.2f} c "
            f"{center_x + control:.2f} {center_y - radius:.2f} "
            f"{center_x + radius:.2f} {center_y - control:.2f} "
            f"{center_x + radius:.2f} {center_y:.2f} c f")

    def text(self, x, y, value, size=13, anchor="start", bold=False, rotate=False):
        value = pdf_text(value)
        estimated_width = len(value) * size * 0.52
        font = "/F2" if bold else "/F1"
        if rotate:
            baseline = self.y(y) - estimated_width / 2.0
            matrix = f"0 1 -1 0 {x:.2f} {baseline:.2f} Tm"
        else:
            if anchor == "middle":
                x -= estimated_width / 2.0
            elif anchor == "end":
                x -= estimated_width
            matrix = f"1 0 0 1 {x:.2f} {self.y(y):.2f} Tm"
        self.commands.append(
            f"BT 0 0 0 rg {font} {size:.2f} Tf {matrix} ({value}) Tj ET")

    def save(self, path):
        content = ("\n".join(self.commands) + "\n").encode("ascii")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {WIDTH} {HEIGHT}] "
             "/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> "
             "/Contents 6 0 R >>").encode("ascii"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
            f"<< /Length {len(content)} >>\nstream\n".encode("ascii") + content + b"endstream",
        ]

        document = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for number, obj in enumerate(objects, start=1):
            offsets.append(len(document))
            document.extend(f"{number} 0 obj\n".encode("ascii"))
            document.extend(obj)
            document.extend(b"\nendobj\n")

        cross_reference = len(document)
        document.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        document.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            document.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        document.extend(
            (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
             f"startxref\n{cross_reference}\n%%EOF\n").encode("ascii"))
        Path(path).write_bytes(document)


def nice_bounds(values, include_zero=True):
    low = min(values)
    high = max(values)
    if include_zero:
        low = min(0.0, low)
        high = max(0.0, high)
    if math.isclose(low, high):
        high = low + (abs(low) if low else 1.0)
    span = high - low
    raw_step = span / 5.0
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    step = next(candidate * magnitude for candidate in (1, 2, 5, 10)
                if candidate * magnitude >= raw_step)
    low = math.floor(low / step) * step
    high = math.ceil(high / step) * step
    return low, high, step


def line_plot(path, title, xlabel, ylabel, x_values, series, x_labels=None):
    all_y = [value for _, points in series for _, value in points]
    y_low, y_high, y_step = nice_bounds(all_y)
    x_low = min(x_values)
    x_high = max(x_values)
    if math.isclose(x_low, x_high):
        x_high = x_low + 1.0

    plot_width = WIDTH - LEFT - RIGHT
    plot_height = HEIGHT - TOP - BOTTOM

    def sx(value):
        return LEFT + (value - x_low) * plot_width / (x_high - x_low)

    def sy(value):
        return TOP + (y_high - value) * plot_height / (y_high - y_low)

    canvas = PdfCanvas()
    canvas.text(WIDTH / 2, 34, title, 22, "middle", True)

    tick = y_low
    while tick <= y_high + y_step * 0.25:
        y = sy(tick)
        canvas.line(LEFT, y, WIDTH - RIGHT, y, "#dddddd")
        canvas.text(LEFT - 12, y + 5, f"{tick:g}", 13, "end")
        tick += y_step

    for index, x_value in enumerate(x_values):
        x = sx(x_value)
        label = x_labels[index] if x_labels else f"{x_value:g}"
        canvas.line(x, TOP, x, HEIGHT - BOTTOM, "#dddddd")
        canvas.text(x, HEIGHT - BOTTOM + 24, label, 13, "middle")

    canvas.line(LEFT, TOP, LEFT, HEIGHT - BOTTOM, width=1.5)
    canvas.line(LEFT, HEIGHT - BOTTOM, WIDTH - RIGHT, HEIGHT - BOTTOM, width=1.5)
    canvas.text((LEFT + WIDTH - RIGHT) / 2, HEIGHT - 22, xlabel, 16, "middle")
    canvas.text(24, (TOP + HEIGHT - BOTTOM) / 2, ylabel, 16, rotate=True)

    for series_index, (name, points) in enumerate(series):
        color = COLORS[series_index % len(COLORS)]
        plotted_points = [(sx(x), sy(y)) for x, y in points]
        canvas.polyline(plotted_points, color)
        for x, y in plotted_points:
            canvas.circle(x, y, 5, color)
        legend_x = LEFT + 18 + series_index * 190
        canvas.line(legend_x, 52, legend_x + 30, 52, color, 3)
        canvas.text(legend_x + 38, 57, name, 14)

    canvas.save(path)


def bar_plot(path, title, ylabel, labels, values, annotation):
    _, y_high, y_step = nice_bounds(values)
    plot_width = WIDTH - LEFT - RIGHT
    plot_height = HEIGHT - TOP - BOTTOM

    def sy(value):
        return TOP + (y_high - value) * plot_height / y_high

    canvas = PdfCanvas()
    canvas.text(WIDTH / 2, 34, title, 22, "middle", True)
    canvas.text(WIDTH / 2, 58, annotation, 14, "middle")

    tick = 0.0
    while tick <= y_high + y_step * 0.25:
        y = sy(tick)
        canvas.line(LEFT, y, WIDTH - RIGHT, y, "#dddddd")
        canvas.text(LEFT - 12, y + 5, f"{tick:g}", 13, "end")
        tick += y_step

    slot = plot_width / len(values)
    bar_width = min(180.0, slot * 0.56)
    for index, (label, value) in enumerate(zip(labels, values)):
        center = LEFT + slot * (index + 0.5)
        x = center - bar_width / 2
        y = sy(value)
        height = HEIGHT - BOTTOM - y
        color = COLORS[index % len(COLORS)]
        canvas.rectangle(x, y, bar_width, height, color)
        canvas.text(center, y - 10, f"{value:.3f}", 15, "middle", True)
        canvas.text(center, HEIGHT - BOTTOM + 28, label, 15, "middle")

    canvas.line(LEFT, TOP, LEFT, HEIGHT - BOTTOM, width=1.5)
    canvas.line(LEFT, HEIGHT - BOTTOM, WIDTH - RIGHT, HEIGHT - BOTTOM, width=1.5)
    canvas.text(24, (TOP + HEIGHT - BOTTOM) / 2, ylabel, 16, rotate=True)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hisq-force", help="HISQ-force scaling summary CSV")
    parser.add_argument("--rhmc", help="RHMC timing summary CSV")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    if not (args.hisq_force or args.rhmc):
        parser.error("provide at least one input CSV")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.hisq_force:
        rows = read_rows(args.hisq_force)
        x_values = [float(row["spatial_l"]) for row in rows]
        if len(rows) == 1:
            row = rows[0]
            legacy = float(row["legacy_median_s"])
            recursive = float(row["recursive_median_s"])
            gauge_kind = row.get("gauge_kind", "thermalized")
            annotation = (f'{row["spatial_l"]}^3 x {row["nt"]}; '
                          f'{gauge_kind} gauge; '
                          f'speedup {float(row["speedup"]):.4f}x; '
                          f'time reduction {float(row["reduction_percent"]):.2f}%')
            path = output_dir / "hisq_force_timing_gain.pdf"
            bar_plot(path, "HISQ-force timing on thermalized configuration",
                     "Median TestForce time (s)", ("Legacy", "Recursive"),
                     (legacy, recursive), annotation)
            print(path)
        else:
            labels = [f'{row["spatial_l"]}^3x{row["nt"]}' for row in rows]
            legacy = [(x, float(row["legacy_median_s"])) for x, row in zip(x_values, rows)]
            recursive = [(x, float(row["recursive_median_s"])) for x, row in zip(x_values, rows)]
            reduction = [(x, float(row["reduction_percent"])) for x, row in zip(x_values, rows)]
            line_plot(output_dir / "hisq_force_scaling.pdf",
                      "HISQ-force scaling (52^3x8 thermalized; other sizes random)",
                      "Lattice", "Median force time (s)", x_values,
                      (("Legacy", legacy), ("Recursive", recursive)), labels)
            line_plot(output_dir / "hisq_force_gain.pdf",
                      "HISQ-force gain (52^3x8 thermalized; other sizes random)",
                      "Lattice", "Time reduction (%)", x_values,
                      (("Recursive gain", reduction),), labels)
            print(output_dir / "hisq_force_scaling.pdf")
            print(output_dir / "hisq_force_gain.pdf")

    if args.rhmc:
        row = read_rows(args.rhmc)[0]
        legacy = float(row["legacy_median_s"])
        recursive = float(row["recursive_median_s"])
        annotation = f'Speedup {float(row["speedup"]):.4f}x; time reduction {float(row["reduction_percent"]):.2f}%'
        path = output_dir / "rhmc_timing_gain.pdf"
        bar_plot(path, "Full RHMC trajectory timing", "Median HMC.update() time (s)",
                 ("Legacy", "Recursive"), (legacy, recursive), annotation)
        print(path)

if __name__ == "__main__":
    main()
