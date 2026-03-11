from pathlib import Path

raw_dir = Path("data/raw")

if not raw_dir.exists():
    raise FileNotFoundError(f"No existe la carpeta: {raw_dir}")


def format_size(num_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024


total_samples = 0
total_size = 0

for gesture_dir in sorted([p for p in raw_dir.iterdir() if p.is_dir()]):
    files = list(gesture_dir.glob("*.npz"))
    n_samples = len(files)
    gesture_size = sum(f.stat().st_size for f in files)

    print(f"{gesture_dir.name}: {n_samples} muestras | {format_size(gesture_size)}")

    total_samples += n_samples
    total_size += gesture_size

print(f"\nTotal muestras: {total_samples}")
print(f"Peso total de los datos: {format_size(total_size)} ({total_size} bytes)")