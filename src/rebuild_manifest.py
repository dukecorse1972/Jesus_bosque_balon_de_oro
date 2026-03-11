"""
rebuild_manifest.py
-------------------
Reconstruye data/manifest.jsonl escaneando data/raw/ en disco.

Útil cuando:
  - El manifest tiene entradas huérfanas por muestras borradas.
  - Quieres empezar de cero sin perder las muestras ya capturadas.
  - Copiaste muestras manualmente entre carpetas.

Uso:
    python src/rebuild_manifest.py
    python src/rebuild_manifest.py --data_dir data --raw_subdir raw
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reconstruye manifest.jsonl desde data/raw/")
    p.add_argument("--data_dir",    type=str, default="data",         help="Carpeta raíz de datos")
    p.add_argument("--raw_subdir",  type=str, default="raw",          help="Subcarpeta con las muestras .npz")
    p.add_argument("--manifest",    type=str, default="manifest.jsonl", help="Nombre del manifest de salida")
    p.add_argument("--dry_run",     action="store_true",              help="Solo muestra lo que haría, no escribe nada")
    return p.parse_args()


def parse_class_dir(dirname: str) -> tuple[int, str] | None:
    """
    Parsea nombres de carpeta del tipo '00_HOLA', '02_BUENOS_DIAS', etc.
    Devuelve (class_id, class_name) o None si no coincide.
    """
    m = re.match(r"^(\d+)_(.+)$", dirname)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def main() -> None:
    args = parse_args()

    data_dir   = Path(args.data_dir)
    raw_dir    = data_dir / args.raw_subdir
    manifest_path = data_dir / args.manifest

    if not raw_dir.exists():
        raise SystemExit(f"No existe la carpeta: {raw_dir}")

    # Descubrir carpetas de clase
    class_dirs = sorted(
        [d for d in raw_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name
    )

    if not class_dirs:
        raise SystemExit(f"No hay subcarpetas en {raw_dir}")

    records = []
    class_counts: dict[str, int] = {}

    for class_dir in class_dirs:
        parsed = parse_class_dir(class_dir.name)
        if parsed is None:
            print(f"[SKIP] Carpeta con nombre no reconocido: {class_dir.name!r} (esperado: '00_NOMBRE')")
            continue

        y_id, y_name = parsed
        npz_files = sorted(class_dir.glob("sample_*.npz"))

        if not npz_files:
            print(f"[WARN] Carpeta vacía (sin .npz): {class_dir.name}")
            continue

        for npz_path in npz_files:
            # Ruta relativa desde data_dir, con separadores Unix
            rel_path = npz_path.resolve().relative_to(data_dir.resolve())
            rel_str = str(rel_path).replace("\\", "/")

            records.append({
                "path":   rel_str,
                "y":      y_id,
                "y_name": y_name,
                "meta":   {},
            })

        class_counts[y_name] = len(npz_files)

    if not records:
        raise SystemExit("No se encontraron muestras .npz en ninguna carpeta de clase.")

    # Resumen
    print(f"\nMuestras encontradas por clase:")
    for name, count in sorted(class_counts.items()):
        print(f"  {name:<20} {count:>4} muestras")
    print(f"\nTotal: {len(records)} entradas")
    print(f"Manifest de salida: {manifest_path}")

    if args.dry_run:
        print("\n[DRY RUN] No se ha escrito nada.")
        return

    # Escribir manifest
    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ Manifest reconstruido con {len(records)} entradas → {manifest_path}")


if __name__ == "__main__":
    main()
