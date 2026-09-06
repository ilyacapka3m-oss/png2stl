#!/usr/bin/env python3
"""
PNG -> STL cookie cutter (со сглаживанием + кольцо-основание).
Берёт PNG с объектом, находит его внешний контур, сглаживает его
алгоритмом Чакина и строит выдавленную "рамку" заданной толщины и высоты в STL.
При необходимости добавляет внешнее кольцо-подставку снизу.

Модель автоматически масштабируется так, чтобы её наибольшая сторона
(по контуру) равнялась заданному размеру в мм (аргумент --size, по умолчанию 80).

Пример:
    python png2cutter.py input.png -o cutter.stl --thickness 3 --height 15 --smooth 8
    python png2cutter.py input.png --size 60            # наибольшая сторона = 60 мм
    python png2cutter.py input.png --ring-width 4 --ring-height 5
    python png2cutter.py input.png --no-ring
"""

import argparse
import os
import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
import trimesh


def load_mask(path: str) -> np.ndarray:
    """Читает PNG и возвращает бинарную маску объекта (255 = объект)."""
    # ВАЖНО: cv2.imread не понимает пути с кириллицей на Windows.
    # Читаем файл байтами через numpy + cv2.imdecode — юникод работает везде.
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть файл: {path}")

    # Прозрачность -> маска
    if img.shape[2] == 4:  # есть канал alpha
        mask = (img[:, :, 3] > 0).astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Если объект тёмный на светлом фоне — инвертируем
    if mask.mean() > 127:
        mask = 255 - mask

    # Чистим шум
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def largest_external_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Контур не найден. Проверьте изображение.")
    c = max(contours, key=cv2.contourArea)
    return c


def chikin_smooth(points: np.ndarray, iterations: int = 8,
                  cut_ratio: float = 0.3) -> np.ndarray:
    """
    Сглаживание ломаной по Чакину (corner cutting).
    points — массив Nx2 (float).
    За каждую итерацию каждая точка заменяется средней двух соседних точек,
    сдвинутых к середине на cut_ratio.
    """
    pts = points.copy()
    for _ in range(iterations):
        n = len(pts)
        if n < 3:
            break
        new_pts = np.empty((n, 2), dtype=np.float64)
        for i in range(n):
            prev_pt = pts[(i - 1) % n]
            next_pt = pts[(i + 1) % n]
            # Точки на отрезках [prev->cur] и [cur->next], смещённые к cur
            p1 = prev_pt + cut_ratio * (pts[i] - prev_pt)
            p2 = next_pt + cut_ratio * (pts[i] - next_pt)
            new_pts[i] = 0.5 * (p1 + p2)
        pts = new_pts
    return pts


def contour_to_polygon(c, px_to_mm: float, smooth_iters: int = 8):
    """Конвертирует OpenCV-контур в Shapely Polygon с переводом в мм и сглаживанием."""
    pts = c.reshape(-1, 2).astype(np.float64)
    pts *= px_to_mm                      # перевод пикселей в мм

    # Сглаживание Чакина
    if smooth_iters > 0:
        pts = chikin_smooth(pts, iterations=smooth_iters, cut_ratio=0.3)

    poly = Polygon(pts)
    if not poly.is_valid:
        poly = make_valid(poly)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly


def build_frame(inner_poly: Polygon, thickness_mm: float, height_mm: float,
               ring_width: float = 0.0, ring_height: float = 0.0):
    """
    Строит рамку (стенку) и при необходимости внешнее кольцо-основание.

    Стенка: от inner_poly до inner_poly.buffer(thickness_mm), высота height_mm.
    Кольцо: от внешней границы стенки до +ring_width, высота ring_height,
            расположено снизу (z = 0 .. ring_height).
    """
    # --- Основная стенка ---
    wall_outer_poly = inner_poly.buffer(thickness_mm, join_style=2)
    if isinstance(wall_outer_poly, MultiPolygon):
        wall_outer_poly = max(wall_outer_poly.geoms, key=lambda g: g.area)

    wall_outer_solid = trimesh.creation.extrude_polygon(wall_outer_poly, height_mm)
    inner_solid = trimesh.creation.extrude_polygon(inner_poly, height_mm)
    frame = wall_outer_solid.difference(inner_solid)

    # --- Внешнее кольцо (подставка) ---
    if ring_width > 0 and ring_height > 0:
        ring_inner_poly = wall_outer_poly  # внутренняя граница кольца = внешняя стенка
        ring_outer_poly = inner_poly.buffer(thickness_mm + ring_width, join_style=2)
        if isinstance(ring_outer_poly, MultiPolygon):
            ring_outer_poly = max(ring_outer_poly.geoms, key=lambda g: g.area)

        ring_outer_solid = trimesh.creation.extrude_polygon(ring_outer_poly, ring_height)
        ring_inner_solid = trimesh.creation.extrude_polygon(ring_inner_poly, ring_height)
        ring = ring_outer_solid.difference(ring_inner_solid)

        # extrude_polygon выдавливает от z=0 вверх — кольцо уже снизу.
        # Объединяем со стенкой (union).
        frame = frame.union(ring)

    return frame


def process_one(png_path: str, output_path: str, thickness: float, height: float,
                size: float, smooth_iters: int, ring_width: float, ring_height: float):
    """Обрабатывает один PNG-файл и сохраняет результат в output_path."""
    # 1. Загрузка и маска
    mask = load_mask(png_path)

    # 2. Контур
    c = largest_external_contour(mask)

    # 3. Автомасштаб: наибольшая сторона контура = size мм
    x, y, w, h = cv2.boundingRect(c)
    max_side_px = max(w, h)
    if max_side_px == 0:
        raise ValueError("Контур имеет нулевые габариты. Проверьте изображение.")
    px_to_mm = size / max_side_px
    print(f"Масштаб: {px_to_mm:.4f} мм/пкс "
          f"(наибольшая сторона {max_side_px} пкс -> {size} мм)")

    # 4. Polygon в мм (с сглаживанием)
    inner_poly = contour_to_polygon(c, px_to_mm, smooth_iters=smooth_iters)
    print(f"Контур: {len(inner_poly.exterior.coords)} точек, "
          f"площадь ≈ {inner_poly.area:.1f} мм²")

    # 5. Рамка (стенка) + кольцо
    frame = build_frame(inner_poly, thickness, height,
                        ring_width=ring_width, ring_height=ring_height)

    if ring_width > 0 and ring_height > 0:
        print(f"Кольцо: ширина {ring_width} мм, высота {ring_height} мм (снаружи стенки, снизу)")

    # 6. Сохранение
    frame.export(output_path)
    print(f"Сохранено: {output_path}")
    print(f"  Размер: {frame.extents[0]:.1f} × {frame.extents[1]:.1f} × "
          f"{frame.extents[2]:.1f} мм")


def main():
    parser = argparse.ArgumentParser(
        description="PNG -> STL cookie cutter (со сглаживанием и кольцом-основанием)")
    parser.add_argument("png", nargs="?", default=None,
                        help="Входной PNG-файл ИЛИ папка с PNG-файлами. "
                             "Если не задано — используется папка 'input'")
    parser.add_argument("--input-dir", default="input",
                        help="Папка с PNG для пакетной обработки, если вход "
                             "не указан (по умолчанию input)")
    parser.add_argument("-o", "--output", default="cutter.stl",
                        help="Имя выходного STL в режиме одного файла "
                             "(по умолчанию cutter.stl)")
    parser.add_argument("--output-dir", default="output",
                        help="Папка для STL в режиме пакетной обработки "
                             "(по умолчанию output)")
    parser.add_argument("--thickness", type=float, default=1.0,
                        help="Толщина стенки в мм (по умолчанию 3)")
    parser.add_argument("--height", type=float, default=12.5,
                        help="Высота формы в мм (по умолчанию 15)")
    parser.add_argument("--size", type=float, default=80.0,
                        help="Наибольшая сторона модели в мм (масштаб подгоняется "
                             "автоматически; по умолчанию 80)")
    parser.add_argument("--smooth", type=int, default=8,
                        help="Итерации сглаживания Чакина, 0 = выключить "
                             "(по умолчанию 8)")

    # --- Кольцо-основание (подставка снизу) ---
    ring_group = parser.add_argument_group("Кольцо-основание (подставка снизу)")
    ring_group.add_argument("--ring-width", type=float, default=3.0,
                            help="Ширина внешнего кольца в мм (по умолчанию 3). "
                                 "0 = отключить кольцо")
    ring_group.add_argument("--ring-height", type=float, default=2.0,
                            help="Высота внешнего кольца в мм (по умолчанию 3). "
                                 "0 = отключить кольцо")
    ring_group.add_argument("--no-ring", action="store_true",
                            help="Полностью отключить кольцо-основание")

    args = parser.parse_args()

    # Кольцо-основание: общие параметры для всех файлов
    if args.no_ring:
        ring_w, ring_h = 0.0, 0.0
    else:
        ring_w, ring_h = args.ring_width, args.ring_height

    # --- Определение режима: один файл или папка ---
    target = args.png if args.png is not None else args.input_dir

    if os.path.isdir(target):
        # Пакетная обработка всех *.png из папки
        png_files = sorted(
            f for f in os.listdir(target)
            if f.lower().endswith(".png")
        )
        if not png_files:
            print(f"В папке '{target}' не найдено ни одного .png-файла.")
            return

        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Пакетная обработка: {len(png_files)} файл(ов) из '{target}' -> "
              f"'{args.output_dir}'\n")

        ok, failed = 0, []
        for i, name in enumerate(png_files, start=1):
            src = os.path.join(target, name)
            out_name = os.path.splitext(name)[0] + ".stl"
            dst = os.path.join(args.output_dir, out_name)
            print(f"[{i}/{len(png_files)}] {name}")
            try:
                process_one(src, dst, args.thickness, args.height,
                            args.size, args.smooth, ring_w, ring_h)
                ok += 1
            except Exception as e:  # не останавливаем всю пачку на одном файле
                print(f"  ОШИБКА: {e}")
                failed.append(name)
            print()

        print("=" * 40)
        print(f"Готово: успешно {ok} из {len(png_files)}")
        if failed:
            print(f"Провалено ({len(failed)}): {', '.join(failed)}")
    else:
        # Режим одного файла
        if not os.path.isfile(target):
            raise FileNotFoundError(f"Не удалось открыть файл: {target}")
        process_one(target, args.output, args.thickness, args.height,
                    args.size, args.smooth, ring_w, ring_h)


if __name__ == "__main__":
    main()
