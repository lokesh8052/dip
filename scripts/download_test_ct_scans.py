
from pathlib import Path
import requests
import zipfile
import sys
import time

# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "raw" / "ct_scans"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://nlst.cancerimagingarchive.net/nbia-api"

TOKEN_URL = f"{BASE_URL}/oauth/token"
METADATA_URL = f"{BASE_URL}/services/v2/getSeriesMetaData"
SIZE_URL = f"{BASE_URL}/services/v2/getSeriesSize"
DOWNLOAD_URL = f"{BASE_URL}/services/v2/getImage"

CHUNK_SIZE = 1024 * 1024
MAX_SCAN_BYTES = 2 * 1024**3
MAX_TOTAL_BYTES = 5 * 1024**3

# ==================================================
# SELECTED TEST PATIENTS
# ==================================================

SCANS = [
    {
        "patient": "128669",
        "label": 1,
        "series_uid": (
            "1.2.840.113654.2.55."
            "90120049509539855929443582076242301386"
        ),
    },
    {
        "patient": "120708",
        "label": 1,
        "series_uid": (
            "1.2.840.113654.2.55."
            "128362675825191379264782186218670385164"
        ),
    },
    {
        "patient": "210559",
        "label": 0,
        "series_uid": (
            "1.3.6.1.4.1.14519.5.2.1.7009.9004."
            "157193169271897322744515721103"
        ),
    },
]


# ==================================================
# AUTHENTICATION
# ==================================================

def get_access_token(session):

    print("Requesting NLST guest token...")

    response = session.post(
        TOKEN_URL,
        data={
            "username": "nbia_guest",
            "password": "",
            "client_id": "NBIA",
            "grant_type": "password",
        },
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(
            f"Authentication failed: HTTP {response.status_code}\n"
            f"{response.text[:2000]}"
        )

    token = response.json().get("access_token")

    if not token:
        raise RuntimeError("No access token returned.")

    print("Authentication successful.")

    return token


# ==================================================
# METADATA VERIFICATION
# ==================================================

def verify_series(session, headers, scan):

    uid = scan["series_uid"]

    print("\nChecking series metadata...")

    response = session.get(
        METADATA_URL,
        params={"SeriesInstanceUID": uid},
        headers=headers,
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(
            f"Metadata request failed: HTTP {response.status_code}\n"
            f"{response.text[:2000]}"
        )

    data = response.json()

    if not isinstance(data, list) or not data:
        raise RuntimeError("No series metadata returned.")

    metadata = data[0]

    returned_uid = metadata.get("Series UID")
    returned_patient = metadata.get("Subject ID")
    returned_collection = metadata.get("Collection")

    if returned_uid != uid:
        raise RuntimeError("Returned Series UID does not match.")

    if str(returned_patient) != scan["patient"]:
        raise RuntimeError(
            f"Patient mismatch: expected {scan['patient']}, "
            f"received {returned_patient}"
        )

    if returned_collection != "NLST":
        raise RuntimeError(
            f"Unexpected collection: {returned_collection}"
        )

    print("Series metadata verified.")
    print("Patient:", returned_patient)
    print("Collection:", returned_collection)
    print("Study date:", metadata.get("Study Date"))
    print("Modality:", metadata.get("Modality"))
    print("Number of images:", metadata.get("Number of Images"))

    return metadata


# ==================================================
# SERIES SIZE
# ==================================================

def get_series_size(session, headers, uid):

    response = session.get(
        SIZE_URL,
        params={"SeriesInstanceUID": uid},
        headers=headers,
        timeout=60,
    )

    if not response.ok:
        print("Series size lookup unavailable:", response.text[:500])
        return None

    try:
        data = response.json()

        if isinstance(data, list) and data:
            data = data[0]

        if isinstance(data, dict):
            size = data.get("TotalSizeInBytes")

            if size is not None:
                return int(size)

    except (ValueError, TypeError):
        pass

    return None


# ==================================================
# STORAGE CHECK
# ==================================================

def existing_storage():

    return sum(
        path.stat().st_size
        for path in OUTPUT_DIR.glob("*.zip")
    )


# ==================================================
# ZIP VALIDATION
# ==================================================

def validate_zip(path):

    if not zipfile.is_zipfile(path):
        return False, 0

    try:
        with zipfile.ZipFile(path) as archive:

            bad_file = archive.testzip()

            if bad_file is not None:
                return False, 0

            file_count = sum(
                1 for name in archive.namelist()
                if not name.endswith("/")
            )

        return file_count > 0, file_count

    except (zipfile.BadZipFile, OSError):
        return False, 0


# ==================================================
# DOWNLOAD ONE PATIENT
# ==================================================

def download_scan(session, headers, scan):

    patient = scan["patient"]
    uid = scan["series_uid"]

    output_file = OUTPUT_DIR / f"patient_{patient}.zip"
    partial_file = OUTPUT_DIR / f"patient_{patient}.zip.part"

    print("\n" + "=" * 55)
    print(f"PATIENT: {patient}")
    print(f"LABEL: {scan['label']}")
    print("=" * 55)

    # Reuse an existing valid ZIP.
    if output_file.exists():

        valid, file_count = validate_zip(output_file)

        if valid:
            print("Valid ZIP already exists. Skipping download.")
            print("Files:", file_count)
            return

        print("Existing ZIP is invalid. Removing it.")
        output_file.unlink()

    # Remove incomplete previous downloads.
    if partial_file.exists():
        partial_file.unlink()

    # Verify the series belongs to the expected patient.
    verify_series(session, headers, scan)

    expected_size = get_series_size(session, headers, uid)

    if expected_size is not None:

        print(
            f"Expected size: "
            f"{expected_size / 1024**2:.2f} MB"
        )

        if expected_size > MAX_SCAN_BYTES:
            raise RuntimeError(
                "Series exceeds the 2 GB safety limit."
            )

    print("\nStarting image download...")

    for attempt in range(1, 4):

        try:

            with session.get(
                DOWNLOAD_URL,
                params={"SeriesInstanceUID": uid},
                headers=headers,
                stream=True,
                timeout=(60, 180),
            ) as response:

                if not response.ok:
                    raise RuntimeError(
                        f"Download HTTP {response.status_code}\n"
                        f"{response.text[:2000]}"
                    )

                downloaded = 0

                with open(partial_file, "wb") as file:

                    for chunk in response.iter_content(
                        chunk_size=CHUNK_SIZE
                    ):

                        if not chunk:
                            continue

                        file.write(chunk)
                        downloaded += len(chunk)

                        if downloaded > MAX_SCAN_BYTES:
                            raise RuntimeError(
                                "Download exceeded 2 GB limit."
                            )

                        if downloaded % (10 * CHUNK_SIZE) < CHUNK_SIZE:

                            print(
                                f"\rDownloaded: "
                                f"{downloaded / 1024**2:.1f} MB",
                                end="",
                                flush=True,
                            )

            print()

            valid, file_count = validate_zip(partial_file)

            if not valid:

                with open(partial_file, "rb") as file:
                    preview = file.read(1000).decode(
                        "utf-8",
                        errors="replace",
                    )

                raise RuntimeError(
                    "Downloaded response is not a valid ZIP.\n"
                    f"Response preview:\n{preview}"
                )

            new_size = partial_file.stat().st_size

            if existing_storage() + new_size > MAX_TOTAL_BYTES:
                raise RuntimeError(
                    "Total ZIP storage would exceed 5 GB."
                )

            partial_file.replace(output_file)

            print(f"\nPatient {patient}: DOWNLOAD SUCCESS")
            print("Files in ZIP:", file_count)

            print(
                f"ZIP size: "
                f"{output_file.stat().st_size / 1024**2:.2f} MB"
            )

            print("Saved:", output_file)

            return

        except Exception as error:

            print(f"\nAttempt {attempt}/3 failed:")
            print(error)

            if partial_file.exists():
                partial_file.unlink()

            if attempt < 3:
                time.sleep(5 * attempt)
            else:
                raise


# ==================================================
# MAIN
# ==================================================

def main():

    session = requests.Session()

    try:

        token = get_access_token(session)

        headers = {
            "Authorization": f"Bearer {token}"
        }

        print(
            f"\nExisting ZIP storage: "
            f"{existing_storage() / 1024**3:.3f} GB"
        )

        # Download all three selected test patients.
        for scan in SCANS:

            download_scan(session, headers, scan)

            print(
                f"\nCurrent ZIP storage: "
                f"{existing_storage() / 1024**3:.3f} GB"
            )

        print("\n" + "=" * 55)
        print("DOWNLOAD SUMMARY")
        print("=" * 55)

        for scan in SCANS:

            patient = scan["patient"]
            path = OUTPUT_DIR / f"patient_{patient}.zip"

            if path.exists():

                valid, count = validate_zip(path)

                print(
                    f"Patient {patient}: "
                    f"{'VALID' if valid else 'INVALID'} | "
                    f"{path.stat().st_size / 1024**2:.2f} MB | "
                    f"{count} files"
                )

            else:
                print(f"Patient {patient}: NOT DOWNLOADED")

        print("\nAll selected downloads processed.")

    except Exception as error:

        print("\nPROCESS STOPPED")
        print(error)
        sys.exit(1)

    finally:
        session.close()


if __name__ == "__main__":
    main()
