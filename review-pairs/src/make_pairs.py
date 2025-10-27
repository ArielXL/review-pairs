import os
import re
import numpy as np
import pandas as pd

from PIL import Image


BASE_DIR = "../../cbis-ddsm-mass"
OUTPUT_XLSX = "../pairs.xlsx"
SAME_SPLIT_ONLY = True

FULL_RE = re.compile(
    r"Mass-(Training|Test)_P_(\d{5})_(LEFT|RIGHT)_(CC|MLO)_FULL_?PRE\.png$",
    re.IGNORECASE,
)
MASK_RE = re.compile(
    r"Mass-(Training|Test)_P_(\d{5})_(LEFT|RIGHT)_(CC|MLO)_MASK_([0-9]+)___?PRE\.png$",
    re.IGNORECASE,
)


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    SUMMARY
    -------
    Normaliza los nombres de las columnas en un DataFrame de pandas.

    PARAMETERS
    ----------
    df : pd.DataFrame
        El DataFrame cuyas columnas se van a normalizar.

    RETURNS
    -------
    pd.DataFrame
        El DataFrame con los nombres de las columnas normalizados.
    """
    df = df.copy()
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    )
    return df


def load_metadata(base_dir: str) -> pd.DataFrame:
    """
    SUMMARY
    -------
    Carga los metadatos de los conjuntos de datos de entrenamiento y prueba.

    PARAMETERS
    ----------
    base_dir : str
        El directorio base donde se encuentran los archivos CSV.

    RETURNS
    -------
    pd.DataFrame
        Un DataFrame que contiene los metadatos de las imágenes.
    """
    train_csv = os.path.join(base_dir, "mass_case_description_train_set.csv")
    test_csv = os.path.join(base_dir, "mass_case_description_test_set.csv")
    dfs = []
    for path, split in [(train_csv, "Train"), (test_csv, "Test")]:
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path)
        df = normalize_cols(df)
        df["split"] = split

        if "patient_id" in df.columns:
            df["patient_id_num"] = (
                df["patient_id"].astype(str).str.extract(r"(\d+)").astype(int)
            )
            df["patient_id_str5"] = df["patient_id_num"].astype(str).str.zfill(5)
        else:
            raise ValueError("No se encontró la columna 'patient_id' en los CSV.")

        lat_col = (
            "left_or_right_breast"
            if "left_or_right_breast" in df.columns
            else "laterality"
        )
        if lat_col not in df.columns:
            raise ValueError(
                "No se encontró la columna de lateralidad ('left_or_right_breast' o 'laterality')."
            )
        df["laterality"] = df[lat_col].astype(str).str.upper().str.strip()

        if "image_view" not in df.columns:
            raise ValueError("No se encontró la columna 'image_view'.")
        df["image_view"] = df["image_view"].astype(str).str.upper().str.strip()

        if "abnormality_id" not in df.columns:
            raise ValueError("No se encontró la columna 'abnormality_id'.")
        df["abnormality_id"] = df["abnormality_id"].astype(int)

        for c in ["pathology", "mass_shape", "mass_margins"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.upper().str.strip()
            else:
                df[c] = np.nan

        df["pathology"] = df["pathology"].replace({"BENIGN_WITHOUT_CALLBACK": "BENIGN"})

        if "breast_density" in df.columns:
            df["breast_density"] = pd.to_numeric(df["breast_density"], errors="coerce")
        else:
            df["breast_density"] = np.nan

        if "assessment" in df.columns:
            df["assessment"] = pd.to_numeric(df["assessment"], errors="coerce")
        else:
            df["assessment"] = np.nan

        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            "No se pudieron cargar los CSV de train/test en la carpeta especificada."
        )

    meta = pd.concat(dfs, ignore_index=True)
    meta["key"] = (
        meta["split"].str.upper()
        + "|"
        + meta["patient_id_str5"]
        + "|"
        + meta["laterality"].str.upper()
        + "|"
        + meta["image_view"].str.upper()
        + "|"
        + meta["abnormality_id"].astype(str)
    )
    return meta


def _normalize_split(split_raw: str) -> str:
    """
    SUMMARY
    -------
    Normaliza el valor de la columna "split" en los metadatos.

    PARAMETERS
    ----------
    split_raw : str
        El valor original de la columna "split".

    RETURNS
    -------
    str
        El valor normalizado de la columna "split".
    """
    return "Train" if split_raw.lower().startswith("train") else "Test"


def parse_full(path: str) -> dict | None:
    """
    SUMMARY
    -------
    Analiza la ruta de una imagen completa y extrae metadatos relevantes.

    PARAMETERS
    ----------
    path : str
        La ruta de la imagen completa.

    RETURNS
    -------
    dict | None
        Un diccionario con los metadatos extraídos o None si no se pudo analizar la ruta.
    """
    m = FULL_RE.search(os.path.basename(path))
    if not m:
        return None
    split_raw, pid, lat, view = m.groups()
    split = _normalize_split(split_raw)
    return {
        "split": split,
        "patient_id_str5": pid,
        "laterality": lat.upper(),
        "image_view": view.upper(),
        "is_full": True,
        "abnormality_id": None,
        "full_path": path,
    }


def parse_mask(path: str) -> dict | None:
    """
    SUMMARY
    -------
    Analiza la ruta de una máscara y extrae metadatos relevantes.

    PARAMETERS
    ----------
    path : str
        La ruta de la máscara.

    RETURNS
    -------
    dict | None
        Un diccionario con los metadatos extraídos o None si no se pudo analizar la ruta.
    """
    m = MASK_RE.search(os.path.basename(path))
    if not m:
        return None
    split_raw, pid, lat, view, abn_id = m.groups()
    split = _normalize_split(split_raw)
    return {
        "split": split,
        "patient_id_str5": pid,
        "laterality": lat.upper(),
        "image_view": view.upper(),
        "is_full": False,
        "abnormality_id": int(abn_id),
        "mask_path": path,
    }


def pathology_from_path(path: str) -> str | None:
    """
    SUMMARY
    -------
    Extrae la información de patología de la ruta de la imagen.

    PARAMETERS
    ----------
    path : str
        La ruta de la imagen.

    RETURNS
    -------
    str | None
        El tipo de patología ("BENIGN" o "MALIGNANT") o None si no se pudo determinar.
    """
    path_up = path.replace("\\", "/").upper()
    if "/BENIGN/" in path_up:
        return "BENIGN"
    if "/MALIGNANT/" in path_up:
        return "MALIGNANT"
    return None


def walk_images(base_dir: str) -> tuple[dict, dict]:
    """
    SUMMARY
    -------
    Recorre el directorio base y crea índices de imágenes completas y máscaras.

    PARAMETERS
    ----------
    base_dir : str
        El directorio base donde buscar imágenes.

    RETURNS
    -------
    tuple[dict, dict]
        Una tupla que contiene dos diccionarios: el índice de imágenes completas y el índice de máscaras.
    """
    full_index = {}
    mask_index = {}
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if not fn.lower().endswith(".png"):
                continue
            path = os.path.join(root, fn)

            info_full = parse_full(path)
            if info_full:
                key_simple = (
                    info_full["split"].upper()
                    + "|"
                    + info_full["patient_id_str5"]
                    + "|"
                    + info_full["laterality"].upper()
                    + "|"
                    + info_full["image_view"].upper()
                )
                full_index[key_simple] = info_full["full_path"]
                continue

            info_mask = parse_mask(path)
            if info_mask:
                key_abn = (
                    info_mask["split"].upper()
                    + "|"
                    + info_mask["patient_id_str5"]
                    + "|"
                    + info_mask["laterality"].upper()
                    + "|"
                    + info_mask["image_view"].upper()
                    + "|"
                    + str(info_mask["abnormality_id"])
                )
                mask_index[key_abn] = info_mask["mask_path"]
                continue

    return full_index, mask_index


def load_mask_size(mask_path: str) -> int:
    """
    SUMMARY
    -------
    Carga la máscara y devuelve su tamaño en píxeles.

    PARAMETERS
    ----------
    mask_path : str
        La ruta de la máscara.

    RETURNS
    -------
    int
        El tamaño de la máscara en píxeles.
    """
    with Image.open(mask_path) as im:
        arr = np.array(im)
    return int(np.count_nonzero(arr))


def build_lesion_table(
    meta: pd.DataFrame, full_index: dict, mask_index: dict
) -> pd.DataFrame:
    """
    SUMMARY
    -------
    Construye una tabla de lesiones a partir de los índices de imágenes completas y máscaras.

    PARAMETERS
    ----------
    meta : pd.DataFrame
        El DataFrame de metadatos que contiene información sobre las imágenes.
    full_index : dict
        El índice de imágenes completas.
    mask_index : dict
        El índice de máscaras.

    RETURNS
    -------
    pd.DataFrame
        Un DataFrame que contiene la tabla de lesiones.
    """
    rows = []
    for key_abn, mask_path in mask_index.items():
        split, pid, lat, view, abn = key_abn.split("|")
        key_simple = "|".join([split, pid, lat, view])
        full_path = full_index.get(key_simple, None)

        mrow = meta.loc[meta["key"] == key_abn]
        if mrow.empty or full_path is None:
            continue
        m = mrow.iloc[0]

        pat_dir = pathology_from_path(full_path)
        pat_csv = (
            str(m.get("pathology", np.nan)).upper()
            if "pathology" in m.index
            else np.nan
        )
        if pat_csv == "BENIGN_WITHOUT_CALLBACK":
            pat_csv = "BENIGN"
        pathology = pat_dir if pat_dir is not None else pat_csv

        try:
            size_px = load_mask_size(mask_path)
        except Exception:
            continue

        rows.append(
            {
                "split": split,
                "patient_id_str5": pid,
                "laterality": lat,
                "image_view": view,
                "abnormality_id": int(abn),
                "full_image": full_path,
                "mask_image": mask_path,
                "size_px": size_px,
                "breast_density": m.get("breast_density", np.nan),
                "mass_shape": m.get("mass_shape", np.nan),
                "pathology": pathology,
                "mass_margins": m.get("mass_margins", np.nan),
                "assessment": m.get("assessment", np.nan),
            }
        )

    lesions = pd.DataFrame(rows)

    for c in [
        "mass_shape",
        "pathology",
        "mass_margins",
        "image_view",
        "laterality",
        "split",
    ]:
        if c in lesions.columns:
            lesions[c] = lesions[c].astype(str).str.upper().str.strip()
    lesions["breast_density"] = pd.to_numeric(
        lesions.get("breast_density"), errors="coerce"
    )
    if "assessment" in lesions.columns:
        lesions["assessment"] = pd.to_numeric(lesions["assessment"], errors="coerce")

    valid_pats = {"BENIGN", "MALIGNANT"}
    lesions = (
        lesions[lesions["pathology"].isin(valid_pats)]
        .dropna(subset=["full_image"])
        .reset_index(drop=True)
    )

    return lesions


def make_pairs(lesions: pd.DataFrame) -> pd.DataFrame:
    """
    SUMMARY
    -------
    Crea pares de imágenes antes y después de la intervención.

    PARAMETERS
    ----------
    lesions : pd.DataFrame
        El DataFrame que contiene la información de las lesiones.

    RETURNS
    -------
    pd.DataFrame
        Un DataFrame que contiene los pares de imágenes.
    """
    group_cols = [
        "breast_density",
        "mass_shape",
        "pathology",
        "image_view",
        "mass_margins",
        "assessment",
    ]

    if SAME_SPLIT_ONLY and "split" in lesions.columns:
        group_cols = ["split"] + group_cols

    needed = group_cols + ["size_px", "full_image"]
    df = lesions.dropna(subset=needed).copy()
    df = df.sort_values(group_cols + ["size_px"]).reset_index(drop=True)

    pairs = []
    for _, g in df.groupby(group_cols, dropna=False):
        g = g.sort_values("size_px").reset_index(drop=True)
        n = len(g)
        for i in range(n):
            size_i = g.loc[i, "size_px"]
            per_cent = 10 * size_i / 100
            candidates = g.loc[(g.index > i) & (g["size_px"] >= size_i + per_cent)]
            if candidates.empty:
                continue

            for j in candidates.index:
                row_i = g.loc[i]
                row_j = g.loc[j]

                # redundante pero seguro
                if row_i["pathology"] != row_j["pathology"]:
                    continue
                if row_i["assessment"] != row_j["assessment"]:
                    continue
                if SAME_SPLIT_ONLY and row_i.get("split") != row_j.get("split"):
                    continue

                pairs.append(
                    {
                        "before_image": row_i["full_image"],
                        "after_image": row_j["full_image"],
                        "before_split": row_i.get("split"),
                        "after_split": row_j.get("split"),
                        "before_image_abnormality_id": int(row_i["abnormality_id"]),
                        "after_image_abnormality_id": int(row_j["abnormality_id"]),
                        "breast_density": row_j["breast_density"],
                        "mass_shape": row_j["mass_shape"],
                        "pathology": row_j["pathology"],
                        "view": row_j["image_view"],
                        "mass_margins": row_j["mass_margins"],
                        "assessment": int(row_j["assessment"]),
                        "size_before_px": int(row_i["size_px"]),
                        "size_after_px": int(row_j["size_px"]),
                        "growth_pct": round(
                            100.0
                            * (row_j["size_px"] - row_i["size_px"])
                            / row_i["size_px"],
                            2,
                        ),
                        "before_image_mask": row_i["mask_image"],
                        "after_image_mask": row_j["mask_image"],
                    }
                )

    return pd.DataFrame(pairs)


def main():
    print("\n🔍 Cargando metadatos...")
    meta = load_metadata(BASE_DIR)

    print("\n📄 Escaneando imágenes y máscaras...")
    full_index, mask_index = walk_images(BASE_DIR)

    print("\n📊 Construyendo tabla de lesiones...")
    lesions = build_lesion_table(meta, full_index, mask_index)
    if lesions.empty:
        raise RuntimeError("No se construyó ninguna lesión válida.")

    print("\n🔗 Formando pares de imágenes...")
    pairs_df = make_pairs(lesions)
    print(f"\n📦 Pares generados: {len(pairs_df)}")

    cols = [
        "before_image",
        "after_image",
        "before_split",
        "after_split",
        "before_image_abnormality_id",
        "after_image_abnormality_id",
        "breast_density",
        "mass_shape",
        "pathology",
        "view",
        "mass_margins",
        "assessment",
        "size_before_px",
        "size_after_px",
        "growth_pct",
        "before_image_mask",
        "after_image_mask",
    ]
    if not pairs_df.empty:
        cols = [c for c in cols if c in pairs_df.columns]
        pairs_df = pairs_df[cols]

    print("\n💾 Guardando archivo excel final...")
    with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
        pairs_df.to_excel(writer, index=False, sheet_name="pairs")

    print("\n✅ Listo.\n")


if __name__ == "__main__":
    main()
