import csv


def save_tracking_csv(tracked_points: list[dict], output_path: str) -> None:
    """Save per-frame tracking data to a CSV file."""
    with open(output_path, "w", newline="") as file_handle:
        print(obj.type() for obj in tracked_points[0])
        writer = csv.DictWriter(file_handle, fieldnames=tracked_points[0].keys())
        writer.writeheader()

        for point in tracked_points:
            if point is not None:
                writer.writerow(point)
            else:
                writer.writerow({"frame": "tracking_failed"})