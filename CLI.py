

def _run_cli() -> None:
    cwd = os.getcwd()

    parser = argparse.ArgumentParser(description="Stabilise a video file.")
    parser.add_argument("--video",    default=os.path.join(cwd, "uploads", "output_001.mp4"), help="Path to source video.")
    parser.add_argument("--output",   default=os.path.join(cwd, "output"),                    help="Base output directory.")
    parser.add_argument("--debug",   default=False, type=bool,                                help="Smoothing radius for stabilisation (in pixels).")
    parser.add_argument("--tiif-output", default=False, type=bool, help="Save tracking data to stacked tiff.")
    parser.add_argument("--chosen-frame",default=0, type=int, help="Chosen frame." ) # make this into slider in the GUI version
    args = parser.parse_args()

    output_dir = os.path.join(args.output, "results_" + time.strftime("%Y%m%d-%H%M%S"))

    print(f"\n► Processing video: {args.video}")
    print(f" Debug Pipeline: {args.debug}")
    start = time.perf_counter()
    result = process_and_stabilize(args.video, output_dir)
    elapsed = time.perf_counter() - start

    print("✓ All done.")
    print(f"  Stabilised video      : {result['stabilized_video']}")
    print(f"  Total processing time : {elapsed:.2f} seconds")

if __name__ == "__main__":
    _run_cli()

