"""경로 설정. 환경변수 FLYBRAIN_DATA 로 데이터 루트를 바꿀 수 있다."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("FLYBRAIN_DATA", ROOT / "data"))

PATHS = {
    "root": ROOT,
    "data": DATA,
    "raw": DATA / "raw",
    "cache": DATA / "cache",
    "results": ROOT / "results",
    # Shiu et al. 2024 (Nature) 가 전처리한 FlyWire v783 연결표 / 뉴런 목록
    "connectivity": DATA / "raw" / "Connectivity_783.parquet",
    "completeness": DATA / "raw" / "Completeness_783.csv",
    # Schlegel et al. 2024 (Nature) 세포 유형 주석 (flyconnectome/flywire_annotations)
    "annotations": DATA / "raw" / "neuron_annotations.tsv",
    # 전처리 캐시
    "cache_npz": DATA / "cache" / "connectome_783.npz",
    "cache_neurons": DATA / "cache" / "neurons_783.parquet",
}

DOWNLOADS = {
    "connectivity": "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Connectivity_783.parquet",
    "completeness": "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Completeness_783.csv",
    "annotations": "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
}
