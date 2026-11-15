
import sys
import io
import time
import gzip
import bz2
import lzma
import hashlib
import socket
import threading
import json
import base64
import http.client
import tempfile

from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# =========================================================
# PATHS AND IMPORTS
# =========================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

from model_inference import load_all_models, predict_patch
from ct_preprocessing import preprocess_mha
from lung_segmentation import create_lung_mask
from scripts.extract_ct_candidates import extract_candidates
from scripts.extract_candidate_patches import extract_candidate_patches

RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

st.set_page_config(
    page_title="DIP-MONet Research Dashboard",
    page_icon="🫁",
    layout="wide"
)

# =========================================================
# GENERAL HELPERS
# =========================================================

def load_csv(filename):
    path = RESULTS_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as error:
        st.warning(f"Could not read {filename}: {error}")
        return pd.DataFrame()


def show_dataframe(df):
    if df.empty:
        st.info("No benchmark data available.")
    else:
        st.dataframe(df, use_container_width=True)


def show_numeric_charts(df, title):
    if df.empty:
        st.info("No data available for plotting.")
        return

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_columns:
        st.info("No numeric columns available.")
        return

    selected_column = st.selectbox(
        f"Select metric for {title}",
        numeric_columns,
        key=f"{title}_metric"
    )

    fig = px.bar(
        df.head(100),
        y=selected_column,
        title=f"{title}: {selected_column}"
    )

    st.plotly_chart(fig, use_container_width=True)


def show_csv_section(title, filename):
    st.subheader(title)

    df = load_csv(filename)

    if df.empty:
        st.warning(f"File missing or empty: {filename}")
        return

    show_dataframe(df)
    show_numeric_charts(df, title)


def format_bytes(value):
    value = float(value)

    for unit in ["B", "KB", "MB", "GB"]:
        if abs(value) < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{value:.2f} TB"


def get_uploaded_data():
    data = st.session_state.get("uploaded_file_bytes")
    name = st.session_state.get("uploaded_file_name")

    if data is None:
        st.info("Upload a file in the Upload & Inference tab first.")
        return None, None

    return data, name


def clear_live_results():
    keys = [
        "patch_predictions",
        "live_hash_results",
        "live_compression_results",
        "live_serialization_results",
        "live_tcp_results",
        "live_http_results",
        "mha_results",
        "mha_scan_info"
    ]

    for key in keys:
        st.session_state.pop(key, None)


# =========================================================
# NPY PATCH LOADER
# =========================================================

def extract_patch(uploaded_bytes):
    data = np.load(
        io.BytesIO(uploaded_bytes),
        allow_pickle=False
    )

    if not isinstance(data, np.ndarray):
        raise ValueError("Uploaded file is not a valid NumPy array.")

    if data.dtype == object:
        raise ValueError("Object arrays are not accepted.")

    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("Patch must contain numeric values.")

    if data.shape != (64, 128, 128):
        raise ValueError(
            f"Expected shape (64, 128, 128), got {data.shape}."
        )

    if not np.isfinite(data).all():
        raise ValueError("Patch contains NaN or infinite values.")

    return data


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def get_models():
    return load_all_models()


# =========================================================
# LIVE HASHING
# =========================================================

def run_hash_benchmark(data):
    algorithms = {
        "SHA-256": lambda x: hashlib.sha256(x),
        "SHA-512": lambda x: hashlib.sha512(x),
        "SHA3-256": lambda x: hashlib.sha3_256(x),
        "BLAKE2b-256": lambda x: hashlib.blake2b(
            x,
            digest_size=32
        )
    }

    rows = []

    for name, algorithm in algorithms.items():
        start = time.perf_counter()
        digest = algorithm(data).hexdigest()
        elapsed = time.perf_counter() - start

        rows.append({
            "Algorithm": name,
            "File Size (bytes)": len(data),
            "Time (seconds)": elapsed,
            "Throughput (MB/s)": (
                len(data) / (1024 * 1024) / elapsed
                if elapsed > 0 else 0
            ),
            "Hash": digest
        })

    return pd.DataFrame(rows)


# =========================================================
# LIVE COMPRESSION
# =========================================================

def run_compression_benchmark(data, filename):
    algorithms = {
        "GZIP": (gzip.compress, gzip.decompress),
        "BZIP2": (bz2.compress, bz2.decompress),
        "LZMA": (lzma.compress, lzma.decompress)
    }

    rows = []

    for name, (compress_function, decompress_function) in algorithms.items():

        start = time.perf_counter()
        compressed = compress_function(data)
        compression_time = time.perf_counter() - start

        start = time.perf_counter()
        restored = decompress_function(compressed)
        decompression_time = time.perf_counter() - start

        rows.append({
            "File": filename,
            "Algorithm": name,
            "Original Size (bytes)": len(data),
            "Compressed Size (bytes)": len(compressed),
            "Compression Ratio": (
                len(data) / len(compressed)
                if len(compressed) else 0
            ),
            "Size Reduction (%)": (
                (1 - len(compressed) / len(data)) * 100
                if len(data) else 0
            ),
            "Compression Time (seconds)": compression_time,
            "Decompression Time (seconds)": decompression_time,
            "Integrity": restored == data
        })

    return pd.DataFrame(rows)


# =========================================================
# LIVE SERIALIZATION
# =========================================================

def run_serialization_benchmark(data):
    rows = []

    # JSON with Base64 representation of uploaded bytes.
    start = time.perf_counter()

    json_data = json.dumps({
        "file_data": base64.b64encode(data).decode("ascii")
    }).encode("utf-8")

    json_time = time.perf_counter() - start

    start = time.perf_counter()

    json_restored = base64.b64decode(
        json.loads(json_data)["file_data"]
    )

    json_decode_time = time.perf_counter() - start

    rows.append({
        "Format": "JSON (Base64)",
        "Serialized Size (bytes)": len(json_data),
        "Serialization Time (seconds)": json_time,
        "Deserialization Time (seconds)": json_decode_time,
        "Integrity": json_restored == data
    })

    # MessagePack.
    try:
        import msgpack

        start = time.perf_counter()

        msgpack_data = msgpack.packb(
            {"file_data": data},
            use_bin_type=True
        )

        msgpack_time = time.perf_counter() - start

        start = time.perf_counter()

        msgpack_restored = msgpack.unpackb(
            msgpack_data,
            raw=False
        )["file_data"]

        msgpack_decode_time = time.perf_counter() - start

        rows.append({
            "Format": "MessagePack",
            "Serialized Size (bytes)": len(msgpack_data),
            "Serialization Time (seconds)": msgpack_time,
            "Deserialization Time (seconds)": msgpack_decode_time,
            "Integrity": msgpack_restored == data
        })

    except ImportError:
        st.warning("Install MessagePack using: pip install msgpack")

    # Protocol Buffers.
    try:
        from google.protobuf.wrappers_pb2 import BytesValue

        start = time.perf_counter()

        protobuf_message = BytesValue(value=data)
        protobuf_data = protobuf_message.SerializeToString()

        protobuf_time = time.perf_counter() - start

        start = time.perf_counter()

        restored_message = BytesValue()
        restored_message.ParseFromString(protobuf_data)

        protobuf_restored = restored_message.value

        protobuf_decode_time = time.perf_counter() - start

        rows.append({
            "Format": "Protocol Buffers (BytesValue)",
            "Serialized Size (bytes)": len(protobuf_data),
            "Serialization Time (seconds)": protobuf_time,
            "Deserialization Time (seconds)": protobuf_decode_time,
            "Integrity": protobuf_restored == data
        })

    except ImportError:
        st.warning("Install Protobuf using: pip install protobuf")

    return pd.DataFrame(rows)


# =========================================================
# LOCAL TCP TRANSFER
# =========================================================

def run_tcp_transfer(data):
    received_data = bytearray()
    server_error = []

    server_socket = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    server_socket.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1
    )

    server_socket.bind(("127.0.0.1", 0))
    server_socket.listen(1)

    port = server_socket.getsockname()[1]

    def receiver():
        try:
            connection, _ = server_socket.accept()

            with connection:
                while True:
                    chunk = connection.recv(1024 * 1024)

                    if not chunk:
                        break

                    received_data.extend(chunk)

        except Exception as error:
            server_error.append(str(error))

        finally:
            server_socket.close()

    thread = threading.Thread(
        target=receiver,
        daemon=True
    )

    thread.start()

    start = time.perf_counter()

    with socket.create_connection(("127.0.0.1", port)) as client:
        client.sendall(data)
        client.shutdown(socket.SHUT_WR)

    thread.join(timeout=120)

    elapsed = time.perf_counter() - start

    if thread.is_alive():
        raise RuntimeError("TCP receiver timed out.")

    if server_error:
        raise RuntimeError(server_error[0])

    received = bytes(received_data)

    return {
        "Protocol": "TCP",
        "Original Size (bytes)": len(data),
        "Received Size (bytes)": len(received),
        "Transfer Time (seconds)": elapsed,
        "Throughput (MB/s)": (
            len(received) / (1024 * 1024) / elapsed
            if elapsed > 0 else 0
        ),
        "Integrity": received == data,
        "Connection": "Localhost"
    }


# =========================================================
# LOCAL HTTP/1.1 TRANSFER
# =========================================================

def run_http_transfer(data):
    received_data = []

    class UploadHandler(BaseHTTPRequestHandler):

        protocol_version = "HTTP/1.1"

        def do_POST(self):
            content_length = int(
                self.headers.get("Content-Length", 0)
            )

            body = self.rfile.read(content_length)
            received_data.append(body)

            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"OK")

        def log_message(self, format, *args):
            pass

    server = HTTPServer(
        ("127.0.0.1", 0),
        UploadHandler
    )

    port = server.server_address[1]

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True
    )

    thread.start()

    start = time.perf_counter()

    try:
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            port,
            timeout=120
        )

        connection.request(
            "POST",
            "/upload",
            body=data,
            headers={
                "Content-Type": "application/octet-stream"
            }
        )

        response = connection.getresponse()
        response.read()

        elapsed = time.perf_counter() - start

        if response.status != 200:
            raise RuntimeError(
                f"HTTP request failed: {response.status}"
            )

        connection.close()

    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    received = received_data[0] if received_data else b""

    return {
        "Protocol": "HTTP/1.1",
        "Original Size (bytes)": len(data),
        "Received Size (bytes)": len(received),
        "Transfer Time (seconds)": elapsed,
        "Throughput (MB/s)": (
            len(received) / (1024 * 1024) / elapsed
            if elapsed > 0 else 0
        ),
        "Integrity": received == data,
        "Connection": "Localhost"
    }


# =========================================================
# FULL CT INFERENCE
# =========================================================

def run_full_ct_inference(ct_bytes, filename, progress_bar, status):

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)

        input_dir = temp_dir / "input"
        processed_dir = temp_dir / "processed"
        mask_dir = temp_dir / "mask"
        candidate_dir = temp_dir / "candidates"
        patch_dir = temp_dir / "patches"

        for directory in [
            input_dir,
            processed_dir,
            mask_dir,
            candidate_dir,
            patch_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        # 1. Save uploaded MHA.
        mha_path = input_dir / Path(filename).name
        mha_path.write_bytes(ct_bytes)

        # 2. Preprocess CT.
        status.write("Step 1/5: Preprocessing CT volume...")

        volume, metadata, ct_path, metadata_path = preprocess_mha(
            mha_path,
            processed_dir
        )

        progress_bar.progress(20)

        # 3. Lung segmentation.
        status.write("Step 2/5: Creating lung mask...")

        lung_mask = create_lung_mask(volume)

        if not np.any(lung_mask):
            raise ValueError(
                "Lung segmentation produced an empty mask."
            )

        mask_path = mask_dir / "lung_mask.npz"

        np.savez_compressed(
            mask_path,
            mask=lung_mask
        )

        progress_bar.progress(40)

        # 4. Candidate extraction.
        status.write("Step 3/5: Extracting CT candidates...")

        extract_candidates(
            ct_path,
            mask_path,
            candidate_dir
        )

        candidates_path = candidate_dir / "candidates.npy"

        if not candidates_path.exists():
            raise FileNotFoundError(
                "Candidate extraction did not create candidates.npy."
            )

        candidates = np.load(
            candidates_path,
            allow_pickle=False
        )

        if len(candidates) == 0:
            raise ValueError("No CT candidates were found.")

        progress_bar.progress(55)

        # 5. Extract patches.
        status.write("Step 4/5: Extracting candidate patches...")

        extract_candidate_patches(
            ct_path,
            candidates_path,
            patch_dir,
            top_k=50
        )

        patch_metadata_path = patch_dir / "patch_metadata.npy"

        if not patch_metadata_path.exists():
            raise FileNotFoundError(
                "Candidate patch metadata was not created."
            )

        patch_metadata = np.load(
            patch_metadata_path,
            allow_pickle=False
        )

        if len(patch_metadata) == 0:
            raise ValueError("No candidate patches were generated.")

        progress_bar.progress(65)

        # 6. Load existing models.
        status.write("Step 5/5: Running DIP-MONet and Baseline inference...")

        models = get_models()

        result_rows = []
        total_patches = len(patch_metadata)

        for index, row in enumerate(patch_metadata):

            rank = int(row[0])
            component_id = int(row[1])
            size = int(row[2])
            z = int(row[3])
            y = int(row[4])
            x = int(row[5])

            patch_path = patch_dir / f"candidate_{rank:03d}.npy"

            if not patch_path.exists():
                continue

            patch = np.load(
                patch_path,
                allow_pickle=False
            )

            predictions = predict_patch(
                patch,
                models
            )

            result = {
                "Candidate Rank": rank,
                "Component ID": component_id,
                "Candidate Size": size,
                "Z": z,
                "Y": y,
                "X": x
            }

            for model_name, prediction in predictions.items():

                result[f"{model_name} Class"] = (
                    prediction["predicted_class"]
                )

                result[f"{model_name} Class 0 Score"] = (
                    prediction["class_0_score"]
                )

                result[f"{model_name} Class 1 Score"] = (
                    prediction["class_1_score"]
                )

            result_rows.append(result)

            progress = 65 + int(
                35 * (index + 1) / total_patches
            )

            progress_bar.progress(min(progress, 100))

        if not result_rows:
            raise ValueError(
                "No candidate patches were successfully processed."
            )

        results_df = pd.DataFrame(result_rows)

        if "DIP-MONet Class 1 Score" in results_df.columns:
            results_df = results_df.sort_values(
                by="DIP-MONet Class 1 Score",
                ascending=False
            ).reset_index(drop=True)

        scan_info = {
            "Volume Shape": str(volume.shape),
            "Candidates Found": len(candidates),
            "Patches Processed": len(result_rows),
            "Lung Voxels": int(lung_mask.sum()),
            "Resampled Spacing": str(
                metadata.get(
                    "resampled_spacing_xyz",
                    "Not available"
                )
            )
        }

        return results_df, scan_info


# =========================================================
# HEADER
# =========================================================

st.title("🫁 DIP-MONet Research Dashboard")

st.markdown(
    """
    Research dashboard combining:

    - DIP-MONet and Baseline 3D CNN inference
    - Uploaded CT patch analysis
    - Live Computer Networks benchmarks
    - Historical benchmark comparisons
    """
)

st.divider()

# =========================================================
# TABS
# =========================================================

tabs = st.tabs([
    "Upload & Inference",
    "Overview",
    "Hashing",
    "Compression",
    "TCP Transfer",
    "Serialization",
    "HTTP Protocols",
    "TCP vs HTTP"
])


# =========================================================
# TAB 1: UPLOAD AND INFERENCE
# =========================================================

with tabs[0]:

    st.header("Upload Medical Image")

    upload_type = st.radio(
        "Choose input type",
        ["3D CT Patch (.npy)", "Full CT Volume (.mha)"],
        horizontal=True
    )

    if upload_type == "3D CT Patch (.npy)":

        uploaded_file = st.file_uploader(
            "Upload a CT patch",
            type=["npy"],
            key="patch_uploader"
        )

        if uploaded_file is not None:

            uploaded_bytes = uploaded_file.getvalue()

            current_signature = (
                uploaded_file.name,
                hashlib.sha256(uploaded_bytes).hexdigest()
            )

            if st.session_state.get("upload_signature") != current_signature:
                st.session_state["upload_signature"] = current_signature
                clear_live_results()

            st.session_state["uploaded_file_bytes"] = uploaded_bytes
            st.session_state["uploaded_file_name"] = uploaded_file.name
            st.session_state["uploaded_file_size"] = len(uploaded_bytes)
            st.session_state["uploaded_file_type"] = "npy"

            try:
                patch = extract_patch(uploaded_bytes)

                st.success("CT patch loaded successfully.")

                col1, col2, col3 = st.columns(3)

                col1.metric("Shape", str(patch.shape))
                col2.metric("Data type", str(patch.dtype))
                col3.metric("File size", format_bytes(len(uploaded_bytes)))

                st.subheader("Patch Statistics")

                stat1, stat2, stat3, stat4 = st.columns(4)

                stat1.metric("Minimum", f"{patch.min():.2f}")
                stat2.metric("Maximum", f"{patch.max():.2f}")
                stat3.metric("Mean", f"{patch.mean():.2f}")
                stat4.metric("Std deviation", f"{patch.std():.2f}")

                middle_slice = patch.shape[0] // 2

                fig = px.imshow(
                    patch[middle_slice],
                    color_continuous_scale="gray",
                    title=f"Central Axial Slice - {middle_slice}"
                )

                st.plotly_chart(fig, use_container_width=True)

                if st.button(
                    "Run DIP-MONet and Baseline Models",
                    type="primary"
                ):

                    with st.spinner("Running model inference..."):

                        try:
                            models = get_models()
                            predictions = predict_patch(patch, models)

                            st.session_state["patch_predictions"] = predictions

                        except Exception as error:
                            st.error(f"Inference failed: {error}")

                predictions = st.session_state.get("patch_predictions")

                if predictions:

                    st.divider()
                    st.subheader("Model Prediction Results")

                    model_columns = st.columns(len(predictions))

                    for index, (model_name, result) in enumerate(predictions.items()):

                        with model_columns[index]:

                            st.markdown(f"### {model_name}")

                            if result["predicted_class"] == 1:
                                st.error("Predicted class: 1")
                            else:
                                st.success("Predicted class: 0")

                            st.metric(
                                "Class 0 score",
                                f"{result['class_0_score']:.4f}"
                            )

                            st.metric(
                                "Class 1 score",
                                f"{result['class_1_score']:.4f}"
                            )

                            st.metric(
                                "Embedding dimensions",
                                len(result["embedding"])
                            )

                    comparison = pd.DataFrame([
                        {
                            "Model": name,
                            "Predicted Class": result["predicted_class"],
                            "Class 0 Score": result["class_0_score"],
                            "Class 1 Score": result["class_1_score"]
                        }
                        for name, result in predictions.items()
                    ])

                    st.subheader("Model Comparison")

                    st.dataframe(
                        comparison,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.info(
                        "These are research classification outputs. "
                        "Softmax scores are not necessarily calibrated "
                        "probabilities of cancer."
                    )

                st.divider()

                st.success(
                    "Uploaded file is available to the live benchmark tabs."
                )

                st.write("Uploaded file:", uploaded_file.name)

            except Exception as error:
                st.error(f"Could not process uploaded NPY file: {error}")

    else:

        uploaded_ct = st.file_uploader(
            "Upload a full CT volume",
            type=["mha"],
            key="mha_uploader"
        )

        if uploaded_ct is not None:

            ct_bytes = uploaded_ct.getvalue()

            current_signature = (
                uploaded_ct.name,
                hashlib.sha256(ct_bytes).hexdigest()
            )

            if st.session_state.get("upload_signature") != current_signature:
                st.session_state["upload_signature"] = current_signature
                clear_live_results()

            st.session_state["uploaded_file_bytes"] = ct_bytes
            st.session_state["uploaded_file_name"] = uploaded_ct.name
            st.session_state["uploaded_file_size"] = len(ct_bytes)
            st.session_state["uploaded_file_type"] = "mha"

            st.success("Full CT volume uploaded successfully.")

            st.metric(
                "File size",
                format_bytes(len(ct_bytes))
            )

            st.info(
                "The uploaded CT will be processed using your existing "
                "preprocessing, segmentation, candidate extraction, "
                "and trained models."
            )

            if st.button(
                "Run Full CT Inference",
                type="primary",
                key="run_mha_inference"
            ):

                try:
                    progress_bar = st.progress(0)
                    status = st.empty()

                    with st.spinner("Processing full CT volume..."):

                        results_df, scan_info = run_full_ct_inference(
                            ct_bytes,
                            uploaded_ct.name,
                            progress_bar,
                            status
                        )

                        st.session_state["mha_results"] = results_df
                        st.session_state["mha_scan_info"] = scan_info

                    status.success("Full CT inference completed.")

                except Exception as error:
                    st.error(f"Full CT processing failed: {error}")

            if "mha_scan_info" in st.session_state:

                st.divider()
                st.subheader("CT Processing Summary")

                info = st.session_state["mha_scan_info"]

                cols = st.columns(3)

                cols[0].metric(
                    "Candidates Found",
                    info["Candidates Found"]
                )

                cols[1].metric(
                    "Patches Processed",
                    info["Patches Processed"]
                )

                cols[2].metric(
                    "Lung Voxels",
                    f'{info["Lung Voxels"]:,}'
                )

                st.write("Volume shape:", info["Volume Shape"])
                st.write("Resampled spacing:", info["Resampled Spacing"])

            if "mha_results" in st.session_state:

                st.divider()
                st.subheader("Full CT Candidate Predictions")

                results_df = st.session_state["mha_results"]

                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download CT Inference Results",
                    data=results_df.to_csv(index=False).encode("utf-8"),
                    file_name="mha_inference_results.csv",
                    mime="text/csv"
                )

                st.info(
                    "These are research model outputs, not clinical diagnoses. "
                    "Class 1 scores are not necessarily calibrated cancer probabilities."
                )


# =========================================================
# TAB 2: OVERVIEW
# =========================================================

with tabs[1]:

    st.header("Project Overview")

    st.markdown(
        """
        **DIP-MONet**
        - DIP-based preprocessing
        - 3D CNN classification
        - Baseline model comparison

        **Computer Networks**
        - Hashing
        - Compression
        - Serialization
        - TCP file transfer
        - HTTP protocol comparison
        """
    )

    data, filename = get_uploaded_data()

    if data is not None:

        st.subheader("Current Uploaded File")

        col1, col2 = st.columns(2)

        col1.metric("Filename", filename)
        col2.metric("File size", format_bytes(len(data)))

    else:
        st.info("No file uploaded yet.")

    st.divider()

    st.subheader("Historical Benchmark Files")

    benchmark_files = [
        "hash_benchmark.csv",
        "compression_benchmark.csv",
        "transfer_benchmark.csv",
        "serialization_benchmark.csv",
        "http_benchmark_raw.csv",
        "tcp_http_comparison.csv"
    ]

    columns = st.columns(3)

    for index, filename in enumerate(benchmark_files):

        df = load_csv(filename)

        with columns[index % 3]:
            st.metric(
                filename.replace("_", " ").replace(".csv", "").title(),
                len(df)
            )

    st.divider()

    st.subheader("Model Checkpoints")

    checkpoint_information = []

    for name, path in {
        "DIP-MONet": CHECKPOINT_DIR / "best_model.pth",
        "Baseline": CHECKPOINT_DIR / "baseline_best_model.pth"
    }.items():

        checkpoint_information.append({
            "Model": name,
            "Available": path.exists(),
            "Size": format_bytes(path.stat().st_size)
            if path.exists() else "Missing"
        })

    st.dataframe(
        pd.DataFrame(checkpoint_information),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# TAB 3: HASHING
# =========================================================

with tabs[2]:

    st.header("Hashing Performance")

    data, filename = get_uploaded_data()

    if data is not None:

        st.subheader(f"Live Results: {filename}")

        if st.button("Run Hashing Benchmark", key="run_hash"):

            with st.spinner("Calculating hashes..."):
                st.session_state["live_hash_results"] = (
                    run_hash_benchmark(data)
                )

        if "live_hash_results" in st.session_state:
            show_dataframe(st.session_state["live_hash_results"])

    st.divider()

    st.subheader("Historical Results")

    show_csv_section(
        "Historical Hashing Performance",
        "hash_benchmark.csv"
    )


# =========================================================
# TAB 4: COMPRESSION
# =========================================================

with tabs[3]:

    st.header("Compression Performance")

    data, filename = get_uploaded_data()

    if data is not None:

        st.subheader(f"Live Results: {filename}")

        if st.button("Run Compression Benchmark", key="run_compression"):

            with st.spinner("Running compression algorithms..."):

                st.session_state["live_compression_results"] = (
                    run_compression_benchmark(data, filename)
                )

        if "live_compression_results" in st.session_state:

            live_df = st.session_state["live_compression_results"]

            show_dataframe(live_df)

            if not live_df.empty:

                fig = px.bar(
                    live_df,
                    x="Algorithm",
                    y="Compression Time (seconds)",
                    title="Compression Time for Uploaded File"
                )

                st.plotly_chart(fig, use_container_width=True)

                fig = px.bar(
                    live_df,
                    x="Algorithm",
                    y="Size Reduction (%)",
                    title="Size Reduction for Uploaded File"
                )

                st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Historical Results")

    show_csv_section(
        "Historical Compression Performance",
        "compression_benchmark.csv"
    )


# =========================================================
# TAB 5: TCP TRANSFER
# =========================================================

with tabs[4]:

    st.header("TCP File Transfer")

    data, filename = get_uploaded_data()

    if data is not None:

        st.subheader(f"Live Results: {filename}")

        if st.button("Run TCP Transfer", key="run_tcp"):

            with st.spinner("Transferring file over localhost TCP..."):

                try:
                    st.session_state["live_tcp_results"] = (
                        run_tcp_transfer(data)
                    )

                except Exception as error:
                    st.error(f"TCP benchmark failed: {error}")

        if "live_tcp_results" in st.session_state:

            result = st.session_state["live_tcp_results"]

            show_dataframe(pd.DataFrame([result]))

            st.caption(
                "This measures a localhost TCP transfer, not an external "
                "network transfer."
            )

    st.divider()

    st.subheader("Historical Results")

    show_dataframe(load_csv("transfer_benchmark.csv"))
    show_dataframe(load_csv("transfer_benchmark_raw.csv"))
    show_dataframe(load_csv("transfer_benchmark_previous.csv"))

    st.divider()

    st.subheader("Historical TCP Metadata Transmission")

    show_dataframe(load_csv("metadata_tcp_summary.csv"))


# =========================================================
# TAB 6: SERIALIZATION
# =========================================================

with tabs[5]:

    st.header("Serialization Performance")

    data, filename = get_uploaded_data()

    if data is not None:

        st.subheader(f"Live Results: {filename}")

        if st.button(
            "Run Serialization Benchmark",
            key="run_serialization"
        ):

            with st.spinner("Serializing uploaded file..."):

                try:
                    st.session_state["live_serialization_results"] = (
                        run_serialization_benchmark(data)
                    )

                except Exception as error:
                    st.error(f"Serialization failed: {error}")

        if "live_serialization_results" in st.session_state:

            show_dataframe(
                st.session_state["live_serialization_results"]
            )

    st.divider()

    st.subheader("Historical Results")

    show_csv_section(
        "Historical Serialization Performance",
        "serialization_benchmark.csv"
    )

    st.subheader("Historical Raw Measurements")

    show_dataframe(load_csv("serialization_benchmark_raw.csv"))


# =========================================================
# TAB 7: HTTP PROTOCOLS
# =========================================================

with tabs[6]:

    st.header("HTTP Protocol Performance")

    data, filename = get_uploaded_data()

    if data is not None:

        st.subheader(f"Live Results: {filename}")

        if st.button("Run HTTP/1.1 Transfer", key="run_http"):

            with st.spinner("Transferring file over HTTP/1.1..."):

                try:
                    st.session_state["live_http_results"] = (
                        run_http_transfer(data)
                    )

                except Exception as error:
                    st.error(f"HTTP benchmark failed: {error}")

        if "live_http_results" in st.session_state:

            show_dataframe(
                pd.DataFrame([
                    st.session_state["live_http_results"]
                ])
            )

        st.warning(
            "HTTP/2 live testing is not implemented yet. "
            "It requires an HTTP/2-capable server and client."
        )

    st.divider()

    st.subheader("Historical HTTP Results")

    show_dataframe(load_csv("http_benchmark_raw.csv"))

    st.subheader("Historical Concurrent Batch Results")

    show_dataframe(load_csv("http_concurrent_batches.csv"))


# =========================================================
# TAB 8: TCP VS HTTP
# =========================================================

with tabs[7]:

    st.header("TCP vs HTTP Comparison")

    data, filename = get_uploaded_data()

    if data is not None:

        st.subheader(f"Live Results: {filename}")

        tcp_result = st.session_state.get("live_tcp_results")
        http_result = st.session_state.get("live_http_results")

        if tcp_result and http_result:

            live_comparison = pd.DataFrame([
                tcp_result,
                http_result
            ])

            show_dataframe(live_comparison)

            fig = px.bar(
                live_comparison,
                x="Protocol",
                y="Transfer Time (seconds)",
                title="TCP vs HTTP/1.1 Transfer Time"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "These are localhost measurements from separate runs. "
                "They are not universal protocol performance results."
            )

        else:

            st.info(
                "Run TCP Transfer and HTTP/1.1 Transfer first "
                "to generate this comparison."
            )

    st.divider()

    st.subheader("Historical Results")

    comparison_df = load_csv("tcp_http_comparison.csv")

    show_dataframe(comparison_df)

    if not comparison_df.empty:
        show_numeric_charts(comparison_df, "Historical TCP vs HTTP")


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "DIP-MONet Research Dashboard | Research and educational use only"
)
