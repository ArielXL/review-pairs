import os
import time
import glob
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image

from typing import Any

from scipy.ndimage import binary_dilation


PAIRS_XLSX = "./review-pairs/pairs.xlsx"
SHEET_NAME = "pairs"
REVIEW_COL = "review"
REVIEW_VALUES = {"accept": "ACEPTAR", "reject": "CANCELAR"}
MASK_ALPHA = 0.35
REQUIRE_MASK = True


def show_img(
    target: st.delta_generator.DeltaGenerator,
    pil_image: Image.Image,
    caption: str,
    fill: bool = True,
) -> None:
    """
    SUMMARY
    -------
    Muestra una imagen con la API nueva de Streamlit
    (width='stretch'/'content').

    PARAMETERS
    ----------
    target : st.delta_generator.DeltaGenerator
        El objetivo donde se mostrará la imagen.
    pil_image : Image.Image
        La imagen PIL a mostrar.
    caption : str
        El texto de la leyenda para la imagen.
    fill : bool
        Si True, la imagen se ajustará al contenedor (width='stretch').
        Si False, se mostrará en su tamaño original (width='content').
    """
    try:
        width_val = "stretch" if fill else "content"
        target.image(pil_image, caption=caption, width=width_val)
    except TypeError:
        target.image(pil_image, caption=caption, use_container_width=fill)


def load_pairs_from_disk(path: str, sheet_name: str) -> pd.DataFrame:
    """
    SUMMARY
    -------
    Carga un archivo Excel y devuelve un DataFrame con los pares.

    PARAMETERS
    ----------
    path : str
        La ruta al archivo Excel.
    sheet_name : str
        El nombre de la hoja a cargar.

    RETURNS
    -------
    pd.DataFrame
        Un DataFrame con los pares cargados.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encontró el Excel de pares: {path}")
    df = pd.read_excel(path, sheet_name=sheet_name)
    if REVIEW_COL not in df.columns:
        df[REVIEW_COL] = pd.NA
    df[REVIEW_COL] = df[REVIEW_COL].astype("string")
    if (
        "growth_pct" not in df.columns
        and "size_before_px" in df.columns
        and "size_after_px" in df.columns
    ):
        sb = pd.to_numeric(df["size_before_px"], errors="coerce")
        sa = pd.to_numeric(df["size_after_px"], errors="coerce")
        df["growth_pct"] = ((sa - sb) / sb * 100.0).round(2)
    return df


def save_full_df_to_disk(path: str, df: pd.DataFrame) -> None:
    """
    SUMMARY
    -------
    Guarda el DataFrame completo en un archivo Excel.

    PARAMETERS
    ----------
    path : str
        La ruta al archivo Excel.
    df : pd.DataFrame
        El DataFrame a guardar.

    RETURNS
    -------
    None
    """
    with pd.ExcelWriter(
        path, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        df.to_excel(writer, index=False, sheet_name=SHEET_NAME)


def ensure_df_loaded(excel_path: str) -> None:
    """
    SUMMARY
    -------
    Asegura que el DataFrame esté cargado en el estado de la sesión.

    PARAMETERS
    ----------
    excel_path : str
        La ruta al archivo Excel.
    """
    if "df" not in st.session_state or "excel_mtime" not in st.session_state:
        df = load_pairs_from_disk(excel_path, SHEET_NAME)
        st.session_state.df = df
        st.session_state.excel_mtime = os.path.getmtime(excel_path)
    else:
        try:
            current_mtime = os.path.getmtime(excel_path)
            if current_mtime != st.session_state.excel_mtime:
                st.session_state.df = load_pairs_from_disk(excel_path, SHEET_NAME)
                st.session_state.excel_mtime = current_mtime
        except FileNotFoundError:
            pass


def write_and_refresh(path: str) -> None:
    """
    SUMMARY
    -------
    Guarda el DataFrame en disco y actualiza la marca de tiempo.

    PARAMETERS
    ----------
    path : str
        La ruta al archivo Excel.
    """
    save_full_df_to_disk(path, st.session_state.df)
    st.session_state.excel_mtime = os.path.getmtime(path)


def safe_open_image(path: str) -> Image.Image | None:
    """
    SUMMARY
    -------
    Intenta abrir una imagen de forma segura.

    PARAMETERS
    ----------
    path : str
        La ruta a la imagen.

    RETURNS
    -------
    Image.Image | None
        La imagen abierta o None si no se pudo abrir.
    """
    try:
        if not isinstance(path, str) or not os.path.isfile(path):
            return None
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        return None


def _safe_int(val: Any) -> int | None:
    """
    SUMMARY
    -------
    Intenta convertir un valor a un entero de forma segura.

    PARAMETERS
    ----------
    val : Any
        El valor a convertir.

    RETURNS
    -------
    int | None
        El valor convertido a entero o None si no se pudo convertir.
    """
    try:
        if pd.isna(val):
            return None
        return int(val)
    except Exception:
        return val


def first_pending_index(df: pd.DataFrame) -> int:
    """
    SUMMARY
    -------
    Devuelve el índice del primer par en espera.

    PARAMETERS
    ----------
    df : pd.DataFrame
        El DataFrame a analizar.

    RETURNS
    -------
    int
        El índice del primer par en espera.
    """
    pending = df[df[REVIEW_COL].isna()]
    return int(pending.index.min()) if not pending.empty else 0


def get_abn_id(row: pd.Series, prefix: str) -> int | None:
    """
    SUMMARY
    -------
    Devuelve el ID de anormalidad de una fila dada.

    PARAMETERS
    ----------
    row : pd.Series
        La fila del DataFrame.
    prefix : str
        El prefijo a utilizar para buscar el ID.

    RETURNS
    -------
    int | None
        El ID de anormalidad o None si no se encontró.
    """
    for col in (f"{prefix}_abnormality_id", f"{prefix}_image_abnormality_id"):
        if col in row and pd.notna(row[col]):
            try:
                return int(row[col])
            except Exception:
                pass
    return None


def get_mask_path_from_row(row: pd.Series, which: str) -> str | None:
    """
    SUMMARY
    -------
    Devuelve la ruta de la máscara para una fila dada.

    PARAMETERS
    ----------
    row : pd.Series
        La fila del DataFrame.
    which : str
        El prefijo a utilizar para buscar la máscara.

    RETURNS
    -------
    str | None
        La ruta de la máscara o None si no se encontró.
    """
    col_mask = f"{which}_mask"
    if (
        col_mask in row
        and isinstance(row[col_mask], str)
        and os.path.isfile(row[col_mask])
    ):
        return row[col_mask]

    full_path = row.get(f"{which}_image")
    if not isinstance(full_path, str) or not os.path.isfile(full_path):
        return None

    abn_id = get_abn_id(row, prefix=which)
    folder = os.path.dirname(full_path)
    base = os.path.basename(full_path)

    if abn_id is not None:
        direct = os.path.join(folder, base.replace("FULL_PRE", f"MASK_{abn_id}___PRE"))
        if os.path.isfile(direct):
            return direct

    candidates = glob.glob(
        os.path.join(folder, base.replace("FULL_PRE", "MASK_*___PRE"))
    )
    if not candidates:
        candidates = glob.glob(os.path.join(folder, "Mass-*MASK*PRE*.png"))
    if not candidates:
        return None

    if abn_id is not None:
        filtered = [p for p in candidates if f"MASK_{abn_id}_" in os.path.basename(p)]
        if filtered:
            return sorted(filtered)[0]

    return sorted(candidates)[0]


# def overlay_mask_on_image(
#     img_rgb: Image.Image, mask_path: str, alpha: float = MASK_ALPHA
# ) -> Image.Image | None:
#     """
#     SUMMARY
#     -------
#     Superpone una máscara en una imagen dada.

#     PARAMETERS
#     ----------
#     img_rgb : Image.Image
#         La imagen en la que se superpondrá la máscara.
#     mask_path : str
#         La ruta a la máscara que se superpondrá.
#     alpha : float
#         El valor de opacidad de la máscara.

#     RETURNS
#     -------
#     Image.Image | None
#         La imagen resultante con la máscara superpuesta o None si hubo un error.
#     """
#     try:
#         with Image.open(mask_path) as m:
#             maskL = m.convert("L")
#     except Exception:
#         return None

#     if maskL.size != img_rgb.size:
#         maskL = maskL.resize(img_rgb.size, resample=Image.NEAREST)

#     mask_np = np.array(maskL)
#     bin_np = (mask_np > 0).astype(np.uint8) * 255
#     if bin_np.max() == 0:
#         return None

#     mask_bin = Image.fromarray(bin_np)

#     overlay = Image.new("RGBA", img_rgb.size, (255, 0, 0, 0))
#     rgba = Image.new("RGBA", img_rgb.size, (255, 0, 0, int(round(alpha * 255))))
#     rgba.putalpha(mask_bin)
#     overlay = Image.alpha_composite(overlay, rgba)

#     dil = mask_bin.filter(ImageFilter.MaxFilter(size=3))
#     ero = mask_bin.filter(ImageFilter.MinFilter(size=3))
#     edge = ImageChops.difference(dil, ero).convert("L")
#     edge = edge.point(lambda p: 255 if p > 0 else 0)

#     edge_rgba = Image.new("RGBA", img_rgb.size, (255, 0, 0, 0))
#     edge_arr = np.array(edge_rgba)
#     e_np = np.array(edge)
#     edge_arr[e_np == 255] = (255, 0, 0, 255)
#     edge_rgba = Image.fromarray(edge_arr)

#     base = img_rgb.convert("RGBA")
#     out = Image.alpha_composite(base, overlay)
#     out = Image.alpha_composite(out, edge_rgba)
#     return out.convert("RGB")


def overlay_mask_on_image(
    img_rgb: Image.Image,
    mask_path: str,
    *,
    alpha: float | None = None,
    edge_alpha: float | None = None,
    edge_color: tuple[int, int, int] = (255, 0, 0),
    edge_thickness: int = 15,
) -> Image.Image | None:
    """
    SUMMARY
    -------
    Superpone una máscara en una imagen dada, resaltando solo el
    borde de la máscara.

    PARAMETERS
    ----------
    img_rgb : Image.Image
        La imagen en la que se superpondrá la máscara.
    mask_path : str
        La ruta a la máscara que se superpondrá.
    alpha : float | None
        Valor de opacidad para compatibilidad (se ignora si edge_alpha está definido).
    edge_alpha : float | None
        El valor de opacidad del borde de la máscara.
    edge_color : tuple[int, int, int]
        El color del borde en formato RGB.
    edge_thickness : int
        El grosor del borde en píxeles.

    RETURNS
    -------
    Image.Image | None
        La imagen resultante con la máscara superpuesta o None si hubo un error.
    """
    try:
        with Image.open(mask_path) as m:
            maskL = m.convert("L")
    except Exception:
        return None

    if maskL.size != img_rgb.size:
        maskL = maskL.resize(img_rgb.size, resample=Image.NEAREST)

    mask_np = np.array(maskL, dtype=np.uint8)
    if mask_np.max() == 0:
        return None

    # Detectar bordes de forma rápida usando diferencias en ejes x e y
    # (versión rápida del gradiente de Canny sobredimensionado)
    edge = np.zeros_like(mask_np, dtype=np.uint8)
    edge[:-1, :] |= mask_np[:-1, :] != mask_np[1:, :]  # diferencias verticales
    edge[:, :-1] |= mask_np[:, :-1] != mask_np[:, 1:]  # diferencias horizontales

    # Grosor del borde (simple dilatación con convolución rápida)
    if edge_thickness > 1:

        edge = binary_dilation(edge, iterations=edge_thickness).astype(np.uint8)

    # Preparar alpha
    if edge_alpha is None:
        edge_alpha = alpha if alpha is not None else 1.0
    a = int(round(255 * max(0.0, min(1.0, float(edge_alpha)))))

    # Crear overlay RGBA con el borde pintado
    r, g, b = edge_color
    edge_rgba = np.zeros((*edge.shape, 4), dtype=np.uint8)
    edge_rgba[edge > 0] = (r, g, b, a)

    # Componer
    base = img_rgb.convert("RGBA")
    edge_img = Image.fromarray(edge_rgba)
    out = Image.alpha_composite(base, edge_img)
    return out.convert("RGB")


def _mask_valid(mask_path: str | None, full_img: Image.Image | None) -> bool:
    """
    SUMMARY
    -------
    Verifica si una máscara es válida en relación con una imagen completa.

    PARAMETERS
    ----------
    mask_path : str | None
        La ruta a la máscara.
    full_img : Image.Image | None
        La imagen completa.

    RETURNS
    -------
    bool
        True si la máscara es válida, False en caso contrario.
    """
    try:
        if not mask_path or not os.path.isfile(mask_path) or full_img is None:
            return False
        with Image.open(mask_path) as m:
            arr = np.array(m.convert("L"))
        return np.any(arr > 0)
    except Exception:
        return False


def set_review_and_advance(idx: int, value_label: str) -> None:
    """
    SUMMARY
    -------
    Establece la revisión y avanza al siguiente par.

    PARAMETERS
    ----------
    idx : int
        El índice del par a revisar.
    value_label : str
        La etiqueta de revisión a establecer.

    RETURNS
    -------
    None
    """
    excel_path_ss = st.session_state.get("excel_path", PAIRS_XLSX)
    only_pending_ss = st.session_state.get("only_pending", True)

    if st.session_state.busy:
        return
    st.session_state.busy = True
    try:
        st.session_state.df.at[idx, REVIEW_COL] = value_label
        write_and_refresh(excel_path_ss)

        if only_pending_ss:
            new_view_idx = st.session_state.df[
                st.session_state.df[REVIEW_COL].isna()
            ].index.tolist()
        else:
            new_view_idx = st.session_state.df.index.tolist()

        if new_view_idx:
            if idx in new_view_idx:
                pos2 = new_view_idx.index(idx)
                st.session_state.idx = new_view_idx[
                    min(len(new_view_idx) - 1, pos2 + 1)
                ]
            else:
                st.session_state.idx = new_view_idx[0]

        st.toast(f"Guardado: {value_label} (par {idx+1})", icon="💾")
        st.rerun()
    finally:
        time.sleep(0.05)
        st.session_state.busy = False


def _basename_or_same(v: str | Any) -> str | Any:
    """
    SUMMARY
    -------
    Devuelve el nombre base de un archivo o la entrada misma si
    no es una cadena.

    PARAMETERS
    ----------
    v : str | any
        La entrada a procesar.

    RETURNS
    -------
    str | any
        El nombre base del archivo o la entrada misma si no es una cadena.
    """
    return os.path.basename(str(v)) if isinstance(v, str) else v


def run_app():
    """
    SUMMARY
    -------
    Inicia la aplicación Streamlit.
    """
    st.set_page_config(page_title="Revisión de pares CBIS-DDSM", layout="wide")

    st.title("Revisión de pares para CBIS-DDSM")
    st.markdown("---")

    st.sidebar.header("Configuración")
    excel_path = st.sidebar.text_input("Ruta al Excel de pares", value=PAIRS_XLSX)
    only_pending = st.sidebar.checkbox("Mostrar solo pendientes", value=True)

    st.session_state.excel_path = excel_path
    st.session_state.only_pending = only_pending

    try:
        ensure_df_loaded(excel_path)
    except Exception as e:
        st.error(str(e))
        st.stop()

    df = st.session_state.df

    if "idx" not in st.session_state:
        st.session_state.idx = first_pending_index(df) if only_pending else 0

    view_idx = (
        df[df[REVIEW_COL].isna()].index.tolist() if only_pending else df.index.tolist()
    )
    if not view_idx:
        done = df[REVIEW_COL].notna().sum()
        total = len(df)
        st.success(f"No hay pares pendientes. 🎉 Revisados: {done} / {total}")
        st.dataframe(df[[REVIEW_COL]].value_counts(dropna=False).rename("conteo"))
        st.stop()

    if st.session_state.idx not in view_idx:
        st.session_state.idx = view_idx[0]

    pos = view_idx.index(st.session_state.idx)
    st.markdown(
        """
    <style>
    div[data-testid="stHorizontalBlock"] div:has(> div.stButton) { margin-right: .25rem; }
    div[data-testid="stHorizontalBlock"] div:has(> div.stButton):last-child { margin-right: 0; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    b1, b2, b3, b4, _ = st.columns([1, 1, 2, 2, 2], gap="small")
    with b1:
        if st.button("⏮️ Primero", key="nav_first"):
            st.session_state.idx = view_idx[0]
    with b2:
        if st.button("◀️ Anterior", key="nav_prev"):
            st.session_state.idx = view_idx[max(0, pos - 1)]
    with b3:
        if st.button("Siguiente ▶️", key="nav_next"):
            st.session_state.idx = view_idx[min(len(view_idx) - 1, pos + 1)]
    with b4:
        if st.button("Último ⏭️", key="nav_last"):
            st.session_state.idx = view_idx[-1]

    i = st.session_state.idx
    row = df.loc[i]

    before_img_for_check = safe_open_image(row.get("before_image"))
    after_img_for_check = safe_open_image(row.get("after_image"))
    before_mask_path_chk = get_mask_path_from_row(row, which="before")
    after_mask_path_chk = get_mask_path_from_row(row, which="after")
    mask_missing = REQUIRE_MASK and not (
        _mask_valid(before_mask_path_chk, before_img_for_check)
        and _mask_valid(after_mask_path_chk, after_img_for_check)
    )

    st.markdown("---")
    st.markdown(
        f"**Estado actual:** {row.get(REVIEW_COL) if pd.notna(row.get(REVIEW_COL)) else 'SIN DECISIÓN'}"
    )

    if "busy" not in st.session_state:
        st.session_state.busy = False

    disable_actions = REQUIRE_MASK and mask_missing
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1, 1, 2, 3], gap="small")
    with col_btn1:
        if st.button("✅ Aceptar", key=f"btn_accept_{i}", disabled=disable_actions):
            set_review_and_advance(i, REVIEW_VALUES["accept"])
    with col_btn2:
        if st.button("❌ Cancelar", key=f"btn_reject_{i}", disabled=disable_actions):
            set_review_and_advance(i, REVIEW_VALUES["reject"])
    with col_btn3:
        if st.button("🧽 Limpiar decisión", key=f"btn_clear_single_{i}"):
            st.session_state.df.at[i, REVIEW_COL] = pd.NA
            write_and_refresh(st.session_state.excel_path)
            st.toast("Decisión limpiada.", icon="🧼")
            st.rerun()

    if "show_clear_all" not in st.session_state:
        st.session_state.show_clear_all = False

    with col_btn4:
        if not st.session_state.show_clear_all:
            if st.button(
                "🧹 Limpiar todos (solo columna review)", key="btn_clear_all_open"
            ):
                st.session_state.show_clear_all = True
                st.rerun()
        else:
            st.error(
                "Esta acción borrará TODAS las decisiones de la columna 'review' y no se puede deshacer."
            )
            c1, c2 = st.columns([1, 1], gap="small")
            with c1:
                if st.button("Sí, limpiar ahora", key="btn_clear_all_confirm"):
                    st.session_state.df[REVIEW_COL] = pd.NA
                    write_and_refresh(st.session_state.excel_path)
                    st.session_state.show_clear_all = False
                    st.toast("Todas las decisiones fueron limpiadas.", icon="🧼")
                    st.rerun()
            with c2:
                if st.button("Cancelar", key="btn_clear_all_cancel"):
                    st.session_state.show_clear_all = False
                    st.info("Operación cancelada.")
                    st.rerun()

    st.markdown("---")

    total = len(df)
    done = df[REVIEW_COL].notna().sum()
    pending = total - done
    st.subheader(
        f"**Par {st.session_state.idx+1} / {total}**  |  Revisados: {done}  |  Pendientes: **{pending}**"
    )

    i = st.session_state.idx
    row = df.loc[i]

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.subheader("Antes")
        before_path = row.get("before_image")
        im_before = safe_open_image(before_path)
        before_mask_path = get_mask_path_from_row(row, which="before")

        if im_before and before_mask_path and os.path.isfile(before_mask_path):
            im_before_ov = overlay_mask_on_image(
                im_before, before_mask_path, alpha=MASK_ALPHA
            )
            if im_before_ov is not None:
                show_img(
                    col_left,
                    im_before_ov,
                    f"{os.path.basename(str(before_path))} (lesión resaltada)",
                    fill=True,
                )
            else:
                st.error("Máscara de 'antes' vacía o inválida.")
        else:
            st.error("No se encontró máscara válida para 'antes'.")

        st.markdown("**Características (antes)**")
        st.write(
            {
                "Densidad": _safe_int(row.get("breast_density")),
                "Forma": row.get("mass_shape"),
                "Patología": row.get("pathology"),
                "BI-RADS": _safe_int(row.get("assessment")),
                "Vista": row.get("view"),
                "Conjunto": row.get("before_split"),
                "Margen de la masa": row.get("mass_margins"),
                "Tamaño del tumor en píxeles": _safe_int(row.get("size_before_px")),
            }
        )

    with col_right:
        st.subheader("Después")
        after_path = row.get("after_image")
        im_after = safe_open_image(after_path)
        after_mask_path = get_mask_path_from_row(row, which="after")

        if im_after and after_mask_path and os.path.isfile(after_mask_path):
            im_after_ov = overlay_mask_on_image(
                im_after, after_mask_path, alpha=MASK_ALPHA
            )
            if im_after_ov is not None:
                show_img(
                    col_right,
                    im_after_ov,
                    f"{os.path.basename(str(after_path))} (lesión resaltada)",
                    fill=True,
                )
            else:
                st.error("Máscara de 'después' vacía o inválida.")
        else:
            st.error("No se encontró máscara válida para 'después'.")

        st.markdown("**Características (después)**")
        st.write(
            {
                "Densidad": _safe_int(row.get("breast_density")),
                "Forma": row.get("mass_shape"),
                "Patología": row.get("pathology"),
                "BI-RADS": _safe_int(row.get("assessment")),
                "Vista": row.get("view"),
                "Conjunto": row.get("after_split"),
                "Margen de la masa": row.get("mass_margins"),
                "Tamaño del tumor en píxeles": _safe_int(row.get("size_after_px")),
                "Crecimiento (%)": _safe_int(row.get("growth_pct")),
            }
        )

    st.markdown("---")

    with st.expander("Ver tabla completa de pares", expanded=False):
        show_cols = ["before_image", "after_image", REVIEW_COL]
        show_cols = [c for c in df.columns if c in show_cols]

        view_df = st.session_state.df[show_cols].copy()

        for col in ("before_image", "after_image"):
            if col in view_df.columns:
                view_df[col] = view_df[col].apply(_basename_or_same)

        view_df.index = pd.RangeIndex(
            start=1, stop=len(view_df) + 1, step=1, name="Fila"
        )
        st.dataframe(view_df)


def main():
    run_app()


if __name__ == "__main__":
    main()
