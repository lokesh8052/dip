
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent.parent
CT_DIR = ROOT / "data" / "raw" / "ct_scans"

PATIENTS = ["128669", "120708", "210559"]


def extract_patient(patient):
    zip_path = CT_DIR / f"patient_{patient}.zip"
    output_dir = CT_DIR / f"patient_{patient}_dicom"

    print("\n" + "=" * 50)
    print(f"PATIENT: {patient}")
    print("=" * 50)

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")

    # Avoid extracting the same archive twice.
    if output_dir.exists() and any(output_dir.iterdir()):
        print("Extraction folder already contains files. Skipping.")
        print("Folder:", output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:

        bad_file = archive.testzip()

        if bad_file is not None:
            raise RuntimeError(f"Corrupt ZIP member: {bad_file}")

        members = archive.infolist()

        # Protect against unsafe archive paths.
        root = output_dir.resolve()

        for member in members:
            target = (output_dir / member.filename).resolve()

            if target != root and root not in target.parents:
                raise RuntimeError(
                    f"Unsafe path found in ZIP: {member.filename}"
                )

        archive.extractall(output_dir)

    extracted_files = [
        path for path in output_dir.rglob("*")
        if path.is_file()
    ]

    print("Extraction successful.")
    print("Files extracted:", len(extracted_files))
    print("Destination:", output_dir)


def main():
    for patient in PATIENTS:
        extract_patient(patient)

    print("\n" + "=" * 50)
    print("EXTRACTION SUMMARY")
    print("=" * 50)

    for patient in PATIENTS:
        folder = CT_DIR / f"patient_{patient}_dicom"

        count = sum(1 for path in folder.rglob("*") if path.is_file())

        print(f"Patient {patient}: {count} files")


if __name__ == "__main__":
    main()
