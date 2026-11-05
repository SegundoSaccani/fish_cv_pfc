"""Funciones compartidas por los notebooks de longitud total (LT) de peces.

Reutiliza la lógica de `fish_length_cm_wrapper.ipynb` (YOLO-seg -> máscara, ArUco -> escala) y agrega
utilidades comunes: carga de imágenes, calibración px->cm, limpieza de máscara, dibujo y escritura de resultados.
Todas las rutas son relativas a la carpeta del repositorio (donde está este archivo).
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

# ------------------------------------------------------------------ constantes / convenciones del original
RAIZ = Path(__file__).resolve().parent
WEIGHTS = RAIZ / "runs" / "yolo11s-seg_pescado" / "weights" / "yolov11-seg_best.pt"
EXAMPLES_DIR = RAIZ / "examples"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONF = 0.3
IMGSZ = 640
TAMANO_ARUCO_CM = 5.0                      # lado real del marcador (cm)
ARUCO_DICT = cv2.aruco.DICT_4X4_1000       # incluye los IDs de 4X4_50/100/250
ARUCO_ID = 17                              # ID del marcador usado para calibrar

COLOR_MASCARA = (255, 0, 255)              # RGB
COLOR_ARUCO = (0, 155, 0)
COLOR_CAMINO = (255, 0, 0)                 # rojo
COLOR_EXTREMOS = (255, 255, 0)             # amarillo
COLOR_HULL = (255, 165, 0)                 # naranja
COLOR_ALT = (0, 255, 255)                  # cian (segunda medida, p. ej. Feret con apertura)

_MODELS: dict = {}                          # cache de modelos por ruta de pesos


# ------------------------------------------------------------------ modelo, imágenes y calibración
def cargar_modelo(weights=WEIGHTS):
    """Carga (una sola vez) el modelo YOLO-seg indicado y lo devuelve."""
    from ultralytics import YOLO
    key = str(weights)
    if key not in _MODELS:
        _MODELS[key] = YOLO(key)
    return _MODELS[key]


def listar_imagenes(carpeta=EXAMPLES_DIR, n=None):
    """Lista ordenada de imágenes de `carpeta`; si `n` no es None devuelve solo las primeras `n`."""
    paths = sorted(p for p in Path(carpeta).iterdir() if p.suffix.lower() in EXTS)
    return paths if n is None else paths[:n]


def fish_masks(img_bgr, model, conf=CONF, imgsz=IMGSZ):
    """Predice máscaras de instancia. Devuelve (masks, confs); masks: lista de arrays uint8 {0,255} del tamaño de la imagen."""
    h, w = img_bgr.shape[:2]
    res = model.predict(img_bgr, imgsz=imgsz, conf=conf, retina_masks=True, verbose=False)[0]
    if res.masks is None:
        return [], []
    confs = res.boxes.conf.cpu().numpy().tolist()
    masks = []
    for m in res.masks.data.cpu().numpy():
        m = (m > 0.5).astype(np.uint8) * 255
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        masks.append(m)
    return masks, confs


def detect_aruco(img_bgr, aruco_dict=ARUCO_DICT):
    """Busca marcadores ArUco. Devuelve (corners, ids); corners: lista de arrays (4, 2) float32."""
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(aruco_dict), cv2.aruco.DetectorParameters())
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return [], []
    return [c.reshape(4, 2) for c in corners], ids.flatten().tolist()


def px_por_cm_desde_aruco(corners, tamano_cm=TAMANO_ARUCO_CM):
    """Escala (px/cm) a partir de las 4 esquinas (4, 2) del ArUco: promedio de sus 4 lados / lado real en cm."""
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    lados = np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)
    return float(lados.mean() / tamano_cm)


def preparar_imagen(path, model, conf=CONF, imgsz=IMGSZ, aruco_id=ARUCO_ID, tamano_aruco_cm=TAMANO_ARUCO_CM):
    """Lee la imagen, segmenta los peces y calibra la escala con el ArUco `aruco_id`.

    Devuelve dict con: img, masks, confs, aruco_corners, aruco_ids, aruco_idx, px_por_cm, estado.
    `estado` es "ok" o el motivo por el que no se puede medir ("sin_peces", "sin_aruco", "aruco_id_incorrecto")."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"No se pudo leer {path}")
    masks, confs = fish_masks(img, model, conf, imgsz)
    corners, ids = detect_aruco(img)
    out = dict(img=img, masks=masks, confs=confs, aruco_corners=corners, aruco_ids=ids,
               aruco_idx=None, px_por_cm=float("nan"), estado="ok")
    if not corners:
        out["estado"] = "sin_aruco"
    elif aruco_id not in ids:
        out["estado"] = "aruco_id_incorrecto"
    else:
        out["aruco_idx"] = ids.index(aruco_id)
        out["px_por_cm"] = px_por_cm_desde_aruco(corners[out["aruco_idx"]], tamano_aruco_cm)
    if out["estado"] == "ok" and not masks:
        out["estado"] = "sin_peces"
    return out


# ------------------------------------------------------------------ máscara
def limpiar_mascara(mask, kernel_cierre=7):
    """Rellena huecos internos y aplica un cierre morfológico elíptico (mismo preproceso que el esqueleto original)."""
    filled = (binary_fill_holes(mask > 0) * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_cierre, kernel_cierre))
    return cv2.morphologyEx(filled, cv2.MORPH_CLOSE, k)


def apertura(mask, kernel):
    """Apertura morfológica elíptica de tamaño `kernel` px (0/None: devuelve la máscara sin cambios).
    Conserva solo la componente conexa más grande para no dejar restos sueltos."""
    if not kernel:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(kernel), int(kernel)))
    op = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((op > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask            # la apertura borró todo: se conserva la máscara original
    mayor = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((lab == mayor) * 255).astype(np.uint8)


def mayor_contorno(mask):
    """Contorno externo de mayor área de la máscara (array (N, 1, 2) int32)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea)


# ------------------------------------------------------------------ método original (esqueleto), para comparar
_K_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def esqueleto_lantuejoul(mask):
    """Esqueleto morfológico (Lantuéjoul) sobre la máscara limpia."""
    img = limpiar_mascara(mask)
    skel = np.zeros_like(img)
    while cv2.countNonZero(img) > 0:
        eroded = cv2.erode(img, _K_CROSS)
        skel = cv2.bitwise_or(skel, cv2.subtract(img, cv2.dilate(eroded, _K_CROSS)))
        img = eroded
    return skel


def longitud_esqueleto_poly(mask, grado=4, n_samples=2000, margin=0.3):
    """Método ORIGINAL: esqueleto -> polinomio (ejes PCA) -> extremos donde la curva sale del convex hull.
    Devuelve (line (200, 2) en px, longitud_px)."""
    skel = esqueleto_lantuejoul(mask)
    hull = cv2.convexHull(mayor_contorno(mask))
    ys, xs = np.nonzero(skel)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    u, v = vt[0], vt[1]
    t, s = (pts - c) @ u, (pts - c) @ v
    scale = np.abs(t).max() or 1.0
    coef = np.polyfit(t / scale, s, grado)
    my, mx = np.nonzero(mask)
    mt = (np.column_stack([mx, my]) - c) @ u
    tmin, tmax = mt.min(), mt.max()

    def curve(tt):
        tt = np.asarray(tt, dtype=np.float64)
        return c + np.outer(tt, u) + np.outer(np.polyval(coef, tt / scale), v)

    def dentro(p):
        return cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), False) >= 0

    L = tmax - tmin
    ts = np.linspace(tmin - margin * L, tmax + margin * L, n_samples)
    inside = np.array([dentro(p) for p in curve(ts)])
    i0 = int(np.argmin(np.abs(ts - (tmin + tmax) / 2)))
    if not inside[i0]:
        i0 = int(np.argmax(inside))
    ends = []
    for d in (-1, 1):
        i = i0
        while 0 <= i + d < len(ts) and inside[i + d]:
            i += d
        lo, hi = ts[i], ts[min(max(i + d, 0), len(ts) - 1)]
        for _ in range(30):
            mid = (lo + hi) / 2
            if dentro(curve([mid])[0]):
                lo = mid
            else:
                hi = mid
        ends.append(lo)
    line = curve(np.linspace(ends[0], ends[1], 200))
    return line, float(np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1)))


# ------------------------------------------------------------------ dibujo
def _etiqueta_con_fondo(out, textos, anclaje, contour, placed, fs, th, alpha_fondo=0.6):
    """Escribe `textos` (lista de líneas) sobre `out` (RGB, in situ) junto a `anclaje` (x, y) con fondo oscuro
    semitransparente. La caja se ubica del lado exterior del pez (arriba si el anclaje está sobre el centroide, si no
    abajo) y se desplaza verticalmente hasta no solaparse con las cajas de `placed` (se agrega la nueva)."""
    H, W = out.shape[:2]
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0] for t in textos]
    lh = max(s[1] for s in sizes) + 2 * th
    bw, bh = max(s[0] for s in sizes) + 2 * th, lh * len(textos) + th
    M = cv2.moments(contour)
    cy = M["m01"] / M["m00"] if M["m00"] else anclaje[1]
    arriba = anclaje[1] <= cy
    x0 = int(np.clip(anclaje[0] - bw / 2, 0, max(0, W - bw)))
    y0 = anclaje[1] - bh - 3 * th if arriba else anclaje[1] + 3 * th
    paso = -(bh + th) if arriba else (bh + th)
    y0 = int(y0)
    for _ in range(30):
        y0c = int(np.clip(y0, 0, max(0, H - bh)))
        if not any(x0 < px1 and x0 + bw > px0 and y0c < py1 and y0c + bh > py0 for px0, py0, px1, py1 in placed):
            break
        y0 += paso
    y0 = int(np.clip(y0, 0, max(0, H - bh)))
    placed.append((x0, y0, x0 + bw, y0 + bh))
    roi = out[y0:y0 + bh, x0:x0 + bw]
    roi[:] = (roi * (1 - alpha_fondo)).astype(np.uint8)
    for i, t in enumerate(textos):
        cv2.putText(out, t, (x0 + th, y0 + th + (i + 1) * lh - th), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th,
                    cv2.LINE_AA)


def dibujar_resultados(img_bgr, peces, aruco_corners=(), tamano_aruco_cm=TAMANO_ARUCO_CM, alpha=0.35):
    """Devuelve una imagen RGB anotada.

    `peces`: lista de dicts con
      - "contour": contorno (N, 1, 2) int32 para la máscara semitransparente,
      - "capas": lista de dicts {"tipo": "linea"|"poligono"|"puntos", "pts": (N, 2), "color": RGB, "grosor": int opcional},
      - "texto": línea(s) de texto (str o lista de str) que se escriben sobre el pez.
    Además dibuja los ArUco con su tamaño real."""
    out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    th = max(2, round(max(out.shape[:2]) / 400))
    fs = th / 3
    overlay = out.copy()
    for p in peces:
        cv2.drawContours(overlay, [p["contour"]], -1, COLOR_MASCARA, thickness=cv2.FILLED)
    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    placed = []                                  # rectángulos de etiquetas ya ubicadas (anti-solapamiento)
    for p in peces:
        for capa in p.get("capas", []):
            pts = np.round(np.asarray(capa["pts"])).astype(np.int32)
            g = capa.get("grosor", th)
            if capa["tipo"] in ("linea", "poligono"):
                cv2.polylines(out, [pts.reshape(-1, 1, 2)], capa["tipo"] == "poligono", capa["color"], g, cv2.LINE_AA)
            else:
                for q in pts:
                    cv2.circle(out, (int(q[0]), int(q[1])), 3 * th, capa["color"], -1, cv2.LINE_AA)
        textos = p.get("texto", [])
        textos = [textos] if isinstance(textos, str) else textos
        x, y, w, _ = cv2.boundingRect(p["contour"])
        if "label_xy" in p:                      # etiqueta junto a un punto, con fondo y sin pisar a las vecinas
            _etiqueta_con_fondo(out, textos, p["label_xy"], p["contour"], placed, fs, th)
            continue
        for i, txt in enumerate(reversed(textos)):
            (tw, th_txt), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            tx = int(np.clip(x + w / 2 - tw / 2, 0, max(0, out.shape[1] - tw)))
            ty = int(max(y - 2 * th - i * (th_txt + 2 * th), th_txt))
            cv2.putText(out, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
            cv2.putText(out, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, COLOR_MASCARA, th, cv2.LINE_AA)
    for c in aruco_corners:
        cv2.polylines(out, [np.asarray(c).astype(np.int32)], True, COLOR_ARUCO, th)
        txt = f"{tamano_aruco_cm:g} cm"
        (tw, th_txt), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        cx, cy = np.asarray(c).mean(0)
        cv2.putText(out, txt, (int(cx - tw / 2), int(cy + th_txt / 2)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), th, cv2.LINE_AA)
    return out


def guardar_rgb(img_rgb, ruta):
    """Guarda una imagen RGB en `ruta` (crea la carpeta si no existe)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ruta), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def cronometro():
    """Devuelve una función que da los segundos transcurridos desde su creación."""
    t0 = time.perf_counter()
    return lambda: time.perf_counter() - t0
