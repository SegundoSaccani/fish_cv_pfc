"""
Pipeline completo: iNaturalist -> filtro CLIP -> mascaras SAM2 -> dataset de especies con mascara.

Etapas (en orden):
    1. descargar : baja fotos 'research grade' de iNaturalist a  descargadas/<especie>/
    2. filtrar   : CLIP descarta huesos / pez en el agua / sin pez / filete (se MUEVEN a
                   descargadas/_descartadas/<razon>/<especie>/, o se borran si ELIMINAR_DESCARTADAS)
    3. mascaras  : OWL-ViT detecta el pez (1 caja por foto) -> SAM2 genera la mascara -> se
                   valida con metricas y se descartan las malas
    4. dataset   : arma  dataset_especies/<especie>/{images,masks}/  + metadata.csv + clases.json

Uso:
    python fish_dataset_pipeline.py                      # corre todas las etapas en secuencia
    python fish_dataset_pipeline.py --desde mascaras     # corre mascaras y dataset
    python fish_dataset_pipeline.py --solo filtrar       # corre solo el filtro CLIP

Cada etapa es reanudable: guarda su avance en disco y al relanzar se saltea lo ya hecho.
Para rehacer una etapa, borra su archivo de estado (ver rutas en CONFIGURACION).

Requisitos:  pip install requests pillow numpy opencv-python scipy torch transformers ultralytics
"""

import argparse
import csv
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image
from scipy import ndimage as ndi

# ============================ CONFIGURACION ============================

BASE = Path(__file__).resolve().parent
DIR_DESCARGAS = BASE / "descargadas"
DIR_DESCARTADAS = DIR_DESCARGAS / "_descartadas"           # rechazadas por CLIP
DIR_ESTADO = DIR_DESCARGAS / "_estado"                     # checkpoints de cada etapa
DIR_MASCARAS = DIR_DESCARGAS / "_mascaras"                 # mascaras intermedias (aceptadas/)
DIR_DATASET = BASE / "dataset_especies"

# Especies: carpeta -> nombre cientifico a buscar en iNaturalist.
# Los "A / B" de la lista original se separaron en una carpeta por especie.
# "spp." = se busca el genero completo (incluye todas sus especies).
ESPECIES = {
    # Siluriformes
    "Pseudoplatystoma_corruscans": "Pseudoplatystoma corruscans",
    "Pseudoplatystoma_fasciatum": "Pseudoplatystoma fasciatum",
    "Luciopimelodus_pati": "Luciopimelodus pati",
    "Zungaro_jahu": "Zungaro jahu",
    "Pterodoras_granulosus": "Pterodoras granulosus",
    "Pimelodus_maculatus": "Pimelodus maculatus",
    "Ageneiosus_inermis": "Ageneiosus inermis",
    "Ageneiosus_militaris": "Ageneiosus militaris",
    "Trachelyopterus_striatulus": "Trachelyopterus striatulus",
    "Trachelyopterus_galeatus": "Trachelyopterus galeatus",
    # Characiformes
    "Salminus_brasiliensis": "Salminus brasiliensis",
    "Prochilodus_lineatus": "Prochilodus lineatus",
    "Megaleporinus_obtusidens": "Megaleporinus obtusidens",
    "Piaractus_mesopotamicus": "Piaractus mesopotamicus",
    "Hoplias_malabaricus": "Hoplias malabaricus",
    "Serrasalmus_maculatus": "Serrasalmus maculatus",
    "Pygocentrus_nattereri": "Pygocentrus nattereri",
    "Brycon_orbignyanus": "Brycon orbignyanus",
    "Schizodon_nasutus": "Schizodon nasutus",
    "Schizodon_borellii": "Schizodon borellii",
    "Astyanax_spp": "Astyanax",
    # Gymnotiformes
    "Gymnotus_carapo": "Gymnotus carapo",
    "Eigenmannia_virescens": "Eigenmannia virescens",
    # Loricariidae
    "Hypostomus_commersoni": "Hypostomus commersoni",
    "Pterygoplichthys_spp": "Pterygoplichthys",
    "Loricaria_catamarcensis": "Loricaria catamarcensis",
    "Rineloricaria_spp": "Rineloricaria",
    # Pleuronectiformes
    "Achirus_lineatus": "Achirus lineatus",
    "Catathyridium_jenynsii": "Catathyridium jenynsii",
    # Myliobatiformes
    "Potamotrygon_motoro": "Potamotrygon motoro",
    "Potamotrygon_brachyura": "Potamotrygon brachyura",
}

# ---- Etapa 1: descarga ----
API = "https://api.inaturalist.org/v1"
USER_AGENT = "fish-cv-pfc/1.0 (proyecto academico)"
PAUSA_API = 0.5            # seg despues de cada request a la API
PAUSA_IMAGEN = 0.1         # seg despues de cada imagen descargada (bajalo si quieres ir mas rapido)
POR_PAGINA = 200           # maximo permitido por iNaturalist
TAMANO_FOTO = "large"      # small(240) | medium(500) | large(1024) | original
MAX_FOTOS_POR_OBS = None   # None = todas las fotos de cada observacion
MAX_OBS_POR_ESPECIE = None # None = todas las disponibles
REINTENTOS = 5

# ---- Etapa 2: filtro CLIP ----
MODELO_CLIP = "openai/clip-vit-base-patch16"
BATCH_CLIP = 32
# categoria -> descripciones en ingles (se suman las probabilidades de cada categoria)
CATEGORIAS_CLIP = {
    "huesos":        ["a photo of a fish skeleton", "a photo of fish bones"],
    "en_agua":       ["a photo of a fish swimming underwater",
                      "a photo of a fish in a river or an aquarium",
                      "a photo of a fish partially submerged in water"],
    "sin_pez":       ["a photo of a landscape with no fish", "a photo of a person's face"],
    "procesado":     ["a photo of a fish fillet or a cooked fish dish"],
    "mano":          ["a photo of a person holding a fish"],
    "fuera_agua":    ["a photo of a dead fish lying on the ground, a boat deck or a table"],
    "fondo_simple":  ["a photo of a single fish on a plain background"],
}
# Categorias que se descartan (saca de aca las que no quieras filtrar)
CATEGORIAS_DESCARTAR = {"huesos", "en_agua", "sin_pez", "procesado"}
UMBRAL_DESCARTE = 0.50     # se descarta si la prob. sumada de CATEGORIAS_DESCARTAR >= esto
ELIMINAR_DESCARTADAS = False  # True = borra en vez de mover a _descartadas (irreversible)

# ---- Etapa 3: mascaras ----
PESOS_SAM = BASE / "sam2.1_b.pt"
MODELO_OWL = "google/owlvit-base-patch32"   # detector zero-shot que da la caja para SAM2
PROMPT_OWL = "a fish"
OWL_SCORE_MIN = 0.10       # confianza minima de la caja
MAX_PECES = 1              # cajas por foto (1 = solo la de mayor score)
# Validacion de mascara (se descarta si falla alguna)
SAM_SCORE_MIN = 0.85
IOU_CAJA_MIN = 0.50        # IoU entre bbox de la mascara y la caja del detector
FRACCION_IMG = (0.05, 0.95)   # area de mascara / area de imagen
COMPONENTE_MIN = 0.90      # fraccion del area en la componente mas grande
SOLIDEZ_MIN = 0.50         # area / area del casco convexo (descarta mascaras fragmentadas)
BORDE_MAX = 0.15           # fraccion del perimetro de la imagen tocada por la mascara (pez cortado)

# ---- Etapa 4: dataset ----
MIN_IMAGENES_ESPECIE = 30  # solo avisa si una especie queda con menos

EXTENSIONES = {".jpg", ".jpeg", ".png"}
ETAPAS = ["descargar", "filtrar", "mascaras", "dataset"]

# =======================================================================


# ---------------------------- utilidades -----------------------------

def fmt_t(seg):
    seg = int(max(seg, 0))
    return f"{seg // 3600:02d}:{seg % 3600 // 60:02d}:{seg % 60:02d}"


class Progreso:
    """Imprime 'n/total (%) | it/s | ETA' como mucho cada `cada` segundos."""

    def __init__(self, etapa, total, hechos=0, cada=3.0):
        self.etapa, self.total, self.hechos0, self.n = etapa, total, hechos, hechos
        self.cada, self.t0, self.ult = cada, time.time(), 0.0

    def avanzar(self, n=1, detalle="", forzar=False):
        self.n += n
        ahora = time.time()
        if not (forzar or self.n >= self.total or ahora - self.ult >= self.cada):
            return
        self.ult = ahora
        proc = self.n - self.hechos0
        vel = proc / max(ahora - self.t0, 1e-9)
        eta = (self.total - self.n) / vel if vel > 0 else 0
        pct = 100 * self.n / max(self.total, 1)
        print(f"  [{self.etapa}] {self.n}/{self.total} ({pct:5.1f}%) | {vel:5.2f} it/s | "
              f"transcurrido {fmt_t(ahora - self.t0)} | ETA {fmt_t(eta)} | {detalle}", flush=True)


def titulo(num, nombre):
    print(f"\n{'=' * 70}\n  ETAPA {num}/4: {nombre.upper()}\n{'=' * 70}", flush=True)


def listar_imagenes(dir_especie):
    return sorted(f for f in dir_especie.iterdir()
                  if f.is_file() and f.suffix.lower() in EXTENSIONES)


def dirs_especies():
    if not DIR_DESCARGAS.exists():
        return []
    return sorted(d for d in DIR_DESCARGAS.iterdir() if d.is_dir() and not d.name.startswith("_"))


def leer_json(ruta, defecto):
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else defecto


def guardar_json(ruta, datos):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(".tmp")
    tmp.write_text(json.dumps(datos, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(ruta)


def leer_csv(ruta):
    if not ruta.exists():
        return []
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class CsvAppend:
    """CSV que se va escribiendo fila a fila (sobrevive a un Ctrl+C)."""

    def __init__(self, ruta, columnas):
        ruta.parent.mkdir(parents=True, exist_ok=True)
        nuevo = not ruta.exists()
        self.f = open(ruta, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=columnas)
        if nuevo:
            self.w.writeheader()

    def escribir(self, fila):
        self.w.writerow(fila)
        self.f.flush()

    def cerrar(self):
        self.f.close()


# ------------------------ etapa 1: iNaturalist -----------------------

_sesion = requests.Session()
_sesion.headers["User-Agent"] = USER_AGENT


def api_get(ruta, params):
    for intento in range(1, REINTENTOS + 1):
        try:
            r = _sesion.get(f"{API}{ruta}", params=params, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                espera = float(r.headers.get("Retry-After", 30 * intento))
                print(f"  API {r.status_code}, reintento {intento}/{REINTENTOS} en {espera:.0f}s", flush=True)
                time.sleep(espera)
                continue
            r.raise_for_status()
            time.sleep(PAUSA_API)
            return r.json()
        except requests.RequestException as e:
            print(f"  Error de red ({e}), reintento {intento}/{REINTENTOS}", flush=True)
            time.sleep(5 * intento)
    raise RuntimeError(f"La API fallo {REINTENTOS} veces: {ruta} {params}")


def buscar_taxon(nombre):
    """Devuelve el id del taxon cuyo nombre (o sinonimo) coincide exactamente, o None."""
    data = api_get("/taxa", {"q": nombre, "is_active": "true", "per_page": 30})
    q = nombre.lower()
    for t in data["results"]:
        if t["name"].lower() == q:
            return t["id"], t["name"]
    for t in data["results"]:
        if (t.get("matched_term") or "").lower() == q:
            print(f"  '{nombre}' es sinonimo de '{t['name']}' en iNaturalist", flush=True)
            return t["id"], t["name"]
    sugeridos = ", ".join(t["name"] for t in data["results"][:5]) or "ninguno"
    print(f"  AVISO: no encontre '{nombre}' en iNaturalist (cercanos: {sugeridos})", flush=True)
    return None


def params_obs(taxon_id, id_above=0):
    return {
        "taxon_id": taxon_id, "quality_grade": "research", "verifiable": "true",
        "photos": "true", "order_by": "id", "order": "asc", "id_above": id_above,
    }


def descargar_foto(url, destino):
    for intento in range(1, REINTENTOS + 1):
        try:
            r = _sesion.get(url, timeout=60)
            r.raise_for_status()
            tmp = destino.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(destino)
            time.sleep(PAUSA_IMAGEN)
            return True
        except requests.RequestException:
            time.sleep(3 * intento)
    return False


def etapa_descargar():
    titulo(1, "descargar de iNaturalist")
    DIR_DESCARGAS.mkdir(exist_ok=True)
    ruta_estado = DIR_ESTADO / "descarga.json"
    estado = leer_json(ruta_estado, {})

    print("Resolviendo taxones y contando observaciones...", flush=True)
    for carpeta, nombre in ESPECIES.items():
        st = estado.setdefault(carpeta, {"n_obs": 0, "n_fotos": 0, "last_id": 0, "done": False})
        if st["done"] or "total" in st and st.get("taxon_id"):
            continue
        if "taxon_id" not in st:
            res = buscar_taxon(nombre)
            if res is None:
                st.update(taxon_id=None, done=True, total=0)
                continue
            st["taxon_id"] = res[0]
        d = api_get("/observations", {**params_obs(st["taxon_id"]), "per_page": 0})
        st["total"] = d["total_results"]
        print(f"  {carpeta}: {st['total']} observaciones research-grade", flush=True)
        guardar_json(ruta_estado, estado)

    def objetivo(st):
        t = st.get("total", 0)
        return min(t, MAX_OBS_POR_ESPECIE) if MAX_OBS_POR_ESPECIE else t

    total = sum(objetivo(s) for s in estado.values())
    prog = Progreso("DESCARGA", total, hechos=sum(s["n_obs"] for s in estado.values()))
    cred = CsvAppend(DIR_DESCARGAS / "_creditos.csv",
                     ["especie", "archivo", "obs_id", "foto_id", "licencia", "autor", "url_obs"])

    try:
        for i, (carpeta, st) in enumerate(estado.items(), 1):
            if carpeta not in ESPECIES or st["done"]:
                continue
            print(f"\n--- ({i}/{len(estado)}) {carpeta}: {st['n_obs']}/{objetivo(st)} obs ya hechas ---", flush=True)
            dir_esp = DIR_DESCARGAS / carpeta
            dir_esp.mkdir(exist_ok=True)

            while not st["done"]:
                data = api_get("/observations", {**params_obs(st["taxon_id"], st["last_id"]),
                                                 "per_page": POR_PAGINA})
                resultados = data["results"]
                if not resultados:
                    st["done"] = True
                    break
                for obs in resultados:
                    fotos = obs.get("photos", [])[:MAX_FOTOS_POR_OBS]
                    for foto in fotos:
                        url = (foto.get("url") or "").replace("/square.", f"/{TAMANO_FOTO}.")
                        if not url:
                            continue
                        ext = Path(url.split("?")[0]).suffix or ".jpg"
                        archivo = f"{obs['id']}_{foto['id']}{ext}"
                        destino = dir_esp / archivo
                        if not destino.exists() and not descargar_foto(url, destino):
                            print(f"  no pude bajar {url}", flush=True)
                            continue
                        st["n_fotos"] += 1
                        cred.escribir({
                            "especie": carpeta, "archivo": archivo, "obs_id": obs["id"],
                            "foto_id": foto["id"], "licencia": foto.get("license_code") or "all-rights-reserved",
                            "autor": foto.get("attribution", ""),
                            "url_obs": f"https://www.inaturalist.org/observations/{obs['id']}",
                        })
                    st["n_obs"] += 1
                    st["last_id"] = obs["id"]
                    prog.avanzar(1, f"{carpeta} ({st['n_fotos']} fotos)")
                    if MAX_OBS_POR_ESPECIE and st["n_obs"] >= MAX_OBS_POR_ESPECIE:
                        st["done"] = True
                        break
                guardar_json(ruta_estado, estado)
            guardar_json(ruta_estado, estado)
            print(f"  {carpeta}: terminada ({st['n_fotos']} fotos)", flush=True)
    finally:
        cred.cerrar()
        guardar_json(ruta_estado, estado)
    prog.avanzar(0, "fin", forzar=True)
    print("Descarga terminada.", flush=True)


# --------------------------- etapa 2: CLIP ---------------------------

def etapa_filtrar():
    titulo(2, "filtrar con CLIP")
    import torch
    from transformers import CLIPModel, CLIPProcessor

    ruta_csv = DIR_ESTADO / "filtro_clip.csv"
    hechos = {(r["especie"], r["archivo"]) for r in leer_csv(ruta_csv)}

    pendientes = []
    for d in dirs_especies():
        pendientes += [(d.name, f) for f in listar_imagenes(d) if (d.name, f.name) not in hechos]
    print(f"Pendientes: {len(pendientes)} imagenes ({len(hechos)} ya procesadas)", flush=True)
    if not pendientes:
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    print(f"Cargando {MODELO_CLIP} en {dev}...", flush=True)
    modelo = CLIPModel.from_pretrained(MODELO_CLIP, torch_dtype=dtype).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(MODELO_CLIP)

    frases, idx_cat = [], []
    for cat, descs in CATEGORIAS_CLIP.items():
        frases += descs
        idx_cat += [cat] * len(descs)
    cats = list(CATEGORIAS_CLIP)
    texto = proc(text=frases, return_tensors="pt", padding=True).to(dev)
    # matriz frase -> categoria para sumar probabilidades
    suma = torch.zeros(len(frases), len(cats), device=dev, dtype=torch.float32)
    for j, c in enumerate(idx_cat):
        suma[j, cats.index(c)] = 1.0

    def cargar(item):
        try:
            with Image.open(item[1]) as im:
                return proc(images=im.convert("RGB"), return_tensors="pt")["pixel_values"][0], True
        except Exception:
            return torch.zeros(3, 224, 224), False

    cols = ["especie", "archivo", "categoria_top", "p_top", "p_descartar", "decision", "razon"] + \
           [f"p_{c}" for c in cats]
    csv_out = CsvAppend(ruta_csv, cols)
    prog = Progreso("CLIP", len(pendientes))
    n_desc = 0
    try:
        with ThreadPoolExecutor(4) as pool:
            for i in range(0, len(pendientes), BATCH_CLIP):
                lote = pendientes[i:i + BATCH_CLIP]
                cargados = list(pool.map(cargar, lote))
                px = torch.stack([c[0] for c in cargados]).to(dev, dtype)
                with torch.no_grad():
                    out = modelo(input_ids=texto["input_ids"], attention_mask=texto["attention_mask"],
                                 pixel_values=px)
                    probs = out.logits_per_image.float().softmax(dim=1) @ suma     # (B, n_cats)
                for (esp, ruta), (_, ok), p in zip(lote, cargados, probs.cpu().numpy()):
                    top = int(p.argmax())
                    p_desc = float(sum(p[cats.index(c)] for c in CATEGORIAS_DESCARTAR))
                    if not ok:
                        razon = "corrupta"
                    elif p_desc >= UMBRAL_DESCARTE:
                        razon = max(CATEGORIAS_DESCARTAR, key=lambda c: p[cats.index(c)])
                    else:
                        razon = ""
                    if razon:
                        n_desc += 1
                        if ELIMINAR_DESCARTADAS:
                            ruta.unlink()
                        else:
                            dest = DIR_DESCARTADAS / razon / esp
                            dest.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(ruta), dest / ruta.name)
                    csv_out.escribir({
                        "especie": esp, "archivo": ruta.name, "categoria_top": cats[top],
                        "p_top": f"{p[top]:.3f}", "p_descartar": f"{p_desc:.3f}",
                        "decision": "descartada" if razon else "ok", "razon": razon,
                        **{f"p_{c}": f"{p[k]:.3f}" for k, c in enumerate(cats)},
                    })
                prog.avanzar(len(lote), f"descartadas en esta corrida: {n_desc}")
    finally:
        csv_out.cerrar()
    prog.avanzar(0, f"fin, descartadas: {n_desc}", forzar=True)
    print(f"Filtro terminado. Descartadas: {n_desc} -> "
          f"{'borradas' if ELIMINAR_DESCARTADAS else DIR_DESCARTADAS}", flush=True)


# ------------------------ etapa 3: SAM2 + OWL ------------------------

def evaluar_mascara(mask, caja, score):
    """Limpia la mascara (mayor componente + rellena huecos) y devuelve (mascara, metricas, razon)."""
    h, w = mask.shape
    area = int(mask.sum())
    if area == 0:
        return mask, {}, "vacia"
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    mayor = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp_share = stats[mayor, cv2.CC_STAT_AREA] / area
    limpia = ndi.binary_fill_holes(labels == mayor)
    area_l = int(limpia.sum())

    x, y, bw, bh = (stats[mayor, k] for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                              cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    ix = max(0, min(x + bw, caja[2]) - max(x, caja[0]))
    iy = max(0, min(y + bh, caja[3]) - max(y, caja[1]))
    inter = ix * iy
    union = bw * bh + (caja[2] - caja[0]) * (caja[3] - caja[1]) - inter
    iou = inter / union if union > 0 else 0.0

    cont, _ = cv2.findContours(limpia.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = np.vstack(cont)
    solidez = area_l / max(cv2.contourArea(cv2.convexHull(pts)), 1.0)
    borde = (limpia[0].sum() + limpia[-1].sum() + limpia[:, 0].sum() + limpia[:, -1].sum()) / (2 * (h + w))
    frac = area_l / (h * w)

    m = {"sam_score": round(score, 3), "iou_caja": round(float(iou), 3), "frac_img": round(frac, 3),
         "comp_share": round(float(comp_share), 3), "solidez": round(float(solidez), 3),
         "borde": round(float(borde), 3)}
    fallos = []
    if score < SAM_SCORE_MIN: fallos.append("sam_score")
    if iou < IOU_CAJA_MIN: fallos.append("iou_caja")
    if not FRACCION_IMG[0] <= frac <= FRACCION_IMG[1]: fallos.append("tamano")
    if comp_share < COMPONENTE_MIN: fallos.append("fragmentada")
    if solidez < SOLIDEZ_MIN: fallos.append("solidez")
    if borde > BORDE_MAX: fallos.append("cortada_en_borde")
    return limpia, m, "+".join(fallos)


def etapa_mascaras():
    titulo(3, "mascaras SAM2")
    import torch
    from transformers import pipeline
    from ultralytics import SAM

    ruta_csv = DIR_ESTADO / "mascaras.csv"
    hechos = {(r["especie"], r["archivo"]) for r in leer_csv(ruta_csv)}
    pendientes = []
    for d in dirs_especies():
        pendientes += [(d.name, f) for f in listar_imagenes(d) if (d.name, f.name) not in hechos]
    print(f"Pendientes: {len(pendientes)} imagenes ({len(hechos)} ya procesadas)", flush=True)
    if not pendientes:
        return

    dev = 0 if torch.cuda.is_available() else -1
    print(f"Cargando detector {MODELO_OWL} y SAM2 ({PESOS_SAM.name}) en "
          f"{'GPU' if dev == 0 else 'CPU'}...", flush=True)
    detector = pipeline("zero-shot-object-detection", model=MODELO_OWL, device=dev)
    sam = SAM(str(PESOS_SAM))

    cols = ["especie", "archivo", "estado", "razon", "owl_score", "sam_score", "iou_caja",
            "frac_img", "comp_share", "solidez", "borde", "mascara"]
    csv_out = CsvAppend(ruta_csv, cols)
    prog = Progreso("MASCARAS", len(pendientes), cada=5.0)
    ok = mal = 0

    for esp, ruta in pendientes:
        filas = []
        try:
            with Image.open(ruta) as im:
                img = im.convert("RGB")
            dets = detector(img, candidate_labels=[PROMPT_OWL], threshold=OWL_SCORE_MIN)
            dets = sorted(dets, key=lambda d: -d["score"])[:MAX_PECES]
            if not dets:
                filas.append({"estado": "rechazada", "razon": "sin_deteccion"})
            for k, det in enumerate(dets):
                b = det["box"]
                caja = [max(0, b["xmin"]), max(0, b["ymin"]),
                        min(img.width, b["xmax"]), min(img.height, b["ymax"])]
                res = sam(np.array(img)[:, :, ::-1], bboxes=[caja], verbose=False,
                          device=0 if dev == 0 else "cpu")[0]
                mask = res.masks.data[0].cpu().numpy() > 0.5
                score = float(res.boxes.conf[0]) if res.boxes is not None and len(res.boxes) else 1.0
                limpia, met, razon = evaluar_mascara(mask, caja, score)
                aceptada = razon == ""
                sufijo = "" if MAX_PECES == 1 else f"_{k}"
                destino = DIR_MASCARAS / ("aceptadas" if aceptada else "rechazadas") / esp /                     f"{ruta.stem}{sufijo}.png"
                destino.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(destino), limpia.astype(np.uint8) * 255)
                filas.append({**met, "owl_score": f"{det['score']:.3f}",
                              "estado": "aceptada" if aceptada else "rechazada", "razon": razon,
                              "mascara": str(destino.relative_to(BASE))})
        except Exception as e:
            filas = [{"estado": "error", "razon": str(e)[:100]}]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for f in filas:
            csv_out.escribir({**{c: "" for c in cols}, "especie": esp, "archivo": ruta.name, **f})
            ok += f["estado"] == "aceptada"
            mal += f["estado"] != "aceptada"
        prog.avanzar(1, f"{esp} | aceptadas {ok} / rechazadas {mal}")
    csv_out.cerrar()
    prog.avanzar(0, f"fin, aceptadas {ok} / rechazadas {mal}", forzar=True)


# --------------------------- etapa 4: dataset ------------------------

def etapa_dataset():
    titulo(4, "armar dataset")
    filas = [r for r in leer_csv(DIR_ESTADO / "mascaras.csv") if r["estado"] == "aceptada"]
    if not filas:
        print("No hay mascaras aceptadas. Corre antes la etapa 'mascaras'.", flush=True)
        return
    creditos = {(r["especie"], r["archivo"]): r for r in leer_csv(DIR_DESCARGAS / "_creditos.csv")}

    DIR_DATASET.mkdir(exist_ok=True)
    meta, conteo = [], {}
    prog = Progreso("DATASET", len(filas), cada=2.0)
    for r in filas:
        esp, archivo = r["especie"], r["archivo"]
        origen = DIR_DESCARGAS / esp / archivo
        masc = BASE / r["mascara"]
        if not origen.exists() or not masc.exists():
            prog.avanzar(1)
            continue
        (DIR_DATASET / esp / "images").mkdir(parents=True, exist_ok=True)
        (DIR_DATASET / esp / "masks").mkdir(parents=True, exist_ok=True)
        destino_img = DIR_DATASET / esp / "images" / archivo
        destino_mask = DIR_DATASET / esp / "masks" / masc.name
        if not destino_img.exists():
            shutil.copy2(origen, destino_img)
        if not destino_mask.exists():
            shutil.copy2(masc, destino_mask)
        cr = creditos.get((esp, archivo), {})
        meta.append({
            "especie": esp, "imagen": str(destino_img.relative_to(DIR_DATASET)),
            "mascara": str(destino_mask.relative_to(DIR_DATASET)),
            "obs_id": cr.get("obs_id", archivo.split("_")[0]), "licencia": cr.get("licencia", ""),
            "autor": cr.get("autor", ""), "sam_score": r["sam_score"], "frac_img": r["frac_img"],
        })
        conteo[esp] = conteo.get(esp, 0) + 1
        prog.avanzar(1, esp)

    with open(DIR_DATASET / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0]))
        w.writeheader()
        w.writerows(meta)
    clases = sorted(conteo)
    guardar_json(DIR_DATASET / "clases.json", {c: i for i, c in enumerate(clases)})

    print(f"\nDataset en {DIR_DATASET}: {len(meta)} imagenes con mascara, {len(clases)} especies", flush=True)
    for c in clases:
        aviso = "   <-- pocas" if conteo[c] < MIN_IMAGENES_ESPECIE else ""
        print(f"  {c:32s} {conteo[c]:5d}{aviso}", flush=True)
    sin = [c for c in ESPECIES if c not in conteo]
    if sin:
        print(f"Especies sin ninguna imagen valida: {', '.join(sin)}", flush=True)


# ----------------------- chequeo y reparacion de estado ----------------

def _reescribir_csv(ruta, filas, columnas):
    """Reemplaza el CSV (dejando copia .bak del original) de forma atomica."""
    shutil.copy2(ruta, ruta.with_suffix(".csv.bak"))
    tmp = ruta.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)
    tmp.replace(ruta)


def _podar_csv(ruta, conservar, nombre):
    """Quita del CSV las filas para las que conservar(fila) es False. Devuelve cuantas quito."""
    if not ruta.exists():
        return 0
    with open(ruta, newline="", encoding="utf-8") as f:
        lector = csv.DictReader(f)
        columnas, filas = lector.fieldnames, list(lector)
    if not columnas:
        return 0
    buenas = [r for r in filas if conservar(r)]
    quitadas = len(filas) - len(buenas)
    if quitadas:
        _reescribir_csv(ruta, buenas, columnas)
        print(f"  [reparar] {nombre}: {quitadas} filas sin imagen/archivo correspondiente "
              f"eliminadas ({len(buenas)} quedan; copia en {ruta.name}.bak)", flush=True)
    return quitadas


def reparar_estado(etapa):
    """Antes de una etapa, sincroniza los archivos de estado de ella y de las anteriores con las
    imagenes que realmente hay en disco (por si se borraron o movieron imagenes a mano)."""
    print(f"\nChequeando estado antes de '{etapa}'...", flush=True)
    nivel = ETAPAS.index(etapa)

    en_carpeta = {(d.name, f.name) for d in dirs_especies() for f in listar_imagenes(d)}
    en_descartadas = set()
    if DIR_DESCARTADAS.exists():
        for f in DIR_DESCARTADAS.glob("*/*/*"):
            if f.is_file() and f.suffix.lower() in EXTENSIONES:
                en_descartadas.add((f.parent.name, f.name))
    cambios = 0

    # --- etapa 1: descarga.json (contador de fotos) y _creditos.csv ---
    ruta_estado = DIR_ESTADO / "descarga.json"
    estado = leer_json(ruta_estado, {})
    modificado = False
    for carpeta, st in estado.items():
        reales = (sum(1 for e, _ in en_carpeta if e == carpeta)
                  + sum(1 for e, _ in en_descartadas if e == carpeta))
        if st.get("n_fotos", 0) != reales:
            print(f"  [reparar] descarga.json {carpeta}: n_fotos {st.get('n_fotos', 0)} -> {reales}", flush=True)
            st["n_fotos"] = reales
            modificado = True
    if modificado:
        shutil.copy2(ruta_estado, ruta_estado.with_suffix(".json.bak"))
        guardar_json(ruta_estado, estado)
        cambios += 1
    vistos = set()

    def credito_ok(r):
        k = (r["especie"], r["archivo"])
        if k in vistos or (k not in en_carpeta and k not in en_descartadas):
            return False
        vistos.add(k)
        return True
    cambios += bool(_podar_csv(DIR_DESCARGAS / "_creditos.csv", credito_ok, "_creditos.csv"))

    # --- etapa 2: filtro_clip.csv ---
    if nivel >= 1:
        def filtro_ok(r):
            k = (r["especie"], r["archivo"])
            if r["decision"] == "ok":
                return k in en_carpeta
            # descartada: debe seguir en _descartadas (o haberse borrado a proposito) y no estar
            # de vuelta en la carpeta de la especie (si la restauraste, se vuelve a filtrar)
            return k not in en_carpeta and (k in en_descartadas or ELIMINAR_DESCARTADAS)
        cambios += bool(_podar_csv(DIR_ESTADO / "filtro_clip.csv", filtro_ok, "filtro_clip.csv"))

    # --- etapa 3: mascaras.csv ---
    if nivel >= 2:
        def mascara_ok(r):
            if (r["especie"], r["archivo"]) not in en_carpeta:
                return False
            if r["estado"] == "aceptada" and not (BASE / r["mascara"]).exists():
                return False
            return True
        cambios += bool(_podar_csv(DIR_ESTADO / "mascaras.csv", mascara_ok, "mascaras.csv"))

    print("  Estado consistente." if not cambios else "  Estado reparado.", flush=True)


# ------------------------------- main --------------------------------

FUNCIONES = {"descargar": etapa_descargar, "filtrar": etapa_filtrar,
             "mascaras": etapa_mascaras, "dataset": etapa_dataset}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--desde", choices=ETAPAS, help="corre esta etapa y todas las siguientes")
    g.add_argument("--solo", choices=ETAPAS, help="corre unicamente esta etapa")
    args = ap.parse_args()

    if args.solo:
        etapas = [args.solo]
    else:
        etapas = ETAPAS[ETAPAS.index(args.desde):] if args.desde else ETAPAS
    print(f"Etapas a correr: {' -> '.join(etapas)}", flush=True)

    t0 = time.time()
    for e in etapas:
        reparar_estado(e)
        FUNCIONES[e]()
    print(f"\nPipeline terminado en {fmt_t(time.time() - t0)}.", flush=True)


if __name__ == "__main__":
    main()
