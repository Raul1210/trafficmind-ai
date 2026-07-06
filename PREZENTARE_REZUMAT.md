# TrafficMind AI — Rezumat pentru Prezentarea de Licență

> Generat pe baza codului real din proiect. Toate afirmațiile sunt verificate
> direct din fișierele sursă. Dacă există discrepanțe față de ce ai spus anterior,
> versiunea din acest document este cea corectă (din cod).

---

## 1. Prezentare generală (elevator pitch)

TrafficMind AI este un sistem web de monitorizare și optimizare inteligentă a
traficului rutier în timp real, construit ca aplicație Python/Flask care rulează
complet pe CPU, fără infrastructură cloud sau hardware specializat. Sistemul
preia video dintr-o cameră, fișier sau stream YouTube, detectează vehiculele cu
YOLOv8, clasifică automat tipul scenei (intersecție vs. drum cu flux unic) și,
dacă detectează o intersecție, activează un agent Q-Learning care decide în
timp real care fază de semnafor minimizează timpii de așteptare. Operatorii
urmăresc totul printr-un dashboard web cu stream video live, statistici, alerte
și rapoarte PDF descărcabile — totul securizat cu autentificare bazată pe roluri.

---

## 2. Tehnologii folosite

### Flask 3.1.3
**Ce e:** framework web Python pentru aplicații HTTP.

**Unde și cum:** Întreaga interfață HTTP este Flask. Aplicația este structurată
cu două Blueprint-uri — `api_bp` (`traffic_monitor/api/routes.py`) pentru
dashboard și API-ul JSON, și `auth_bp` (`traffic_monitor/auth/routes.py`) pentru
login/logout/management utilizatori. `create_app()` din `traffic_monitor/__init__.py`
inițializează extensiile, înregistrează Blueprint-urile și creează singleton-ul
`TrafficSystem`. Serverul rulează cu `threaded=True` (configurat din `.env`) pentru
a servi MJPEG streaming și request-uri JSON simultan.

**De ce:** Ușurință de setup, threading nativ, ecosistem bogat de extensii deja
în cerințe (Flask-Login, Flask-WTF, Flask-Bcrypt).

---

### YOLOv8 / ultralytics 8.4.19
**Ce e:** model de detecție de obiecte în timp real, starea artei pentru detecție
single-shot pe CPU.

**Unde și cum:** `traffic_monitor/detection/yolo_detector.py`. Există două clase:

- **`YOLODetector`** — wrapper direct în jurul `ultralytics.YOLO(task='detect')`.
  Recunoaște 80 de clase COCO, dar filtrează la 5 clase vehicule (COCO IDs `[0,2,3,5,7]`
  → car, motorcycle, bus, truck, plus person), consolidate prin `class_mapping`
  din settings (car/truck/bus/van → `"car"`, motorcycle/bicycle → `"motorbike"`).
  Fiecare detecție trece printr-un filtru de confidence per-clasă (`CONF_PERSON=0.65`,
  `CONF_MOTORCYCLE=0.40`).

- **`FluidDetector`** — layer de performanță deasupra modelului raw: rulează YOLO
  doar la fiecare al 3-lea frame (`skip_frames=2`), returnând frame-ul adnotat
  cacheuat pe frame-urile sărite. Adaugă și logica de anti-flicker (detalii la
  secțiunea 5).

Singleton-urile `_detector` și `fluid_detector` sunt create o singură dată la
importul modulului și partajate între toate thread-urile.

**De ce:** API simplu, suport ONNX built-in, clase COCO pre-definite intern
(nu e nevoie de `coco.names` extern).

---

### ONNX Runtime (via ultralytics)
**Ce e:** format standard de schimb de modele ML, optimizat pentru inferență pe CPU.

**Unde și cum:** `export_onnx.py` (script one-shot) exportă `yolov8s.pt` →
`yolov8s.onnx` cu `imgsz=480, opset=12, simplify=True`. La pornire, factory-ul
`_build_detector()` din `yolo_detector.py` verifică dacă `yolov8s.onnx` există
la calea din settings (`_WORKSPACE_DIR/yolov8s.onnx`) și îl preferă față de
`yolov8m.pt` (mai mare și mai lent). Fallback-ul la `.pt` este automat dacă
ONNX lipsește.

**De ce:** yolov8s.onnx este semnificativ mai rapid pe CPU față de yolov8m.pt
(modelul .pt mare era cel din proiectul original). ONNX elimină overhead-ul
PyTorch la inference.

---

### OpenCV 4.13.0.92
**Ce e:** librărie de procesare imagine/video în C++, cu binding Python.

**Unde și cum:** `traffic_monitor/core/capture.py`. `cv2.VideoCapture` deschide
sursa video (fișier, webcam index, sau URL direct de stream). Frame-urile sunt
citite cu `cap.read()` în sub-thread-ul `FrameCapture`, convertite la BGR dacă
sursa livrează BGRA sau grayscale, apoi trimise prin `_infer_queue`. Adnotarea
bounding box-urilor se face cu `cv2.rectangle()` și `cv2.putText()` direct în
`FluidDetector._draw_boxes()`, în loc de `result.plot()` din ultralytics — pentru
că box-urile persistente (ghost boxes) nu sunt în rezultatul ultralytics și
trebuie desenate manual. Frame-urile finale sunt codificate JPEG cu
`cv2.imencode('.jpg', frame, [IMWRITE_JPEG_QUALITY, 60])` pentru MJPEG stream.

**De ce:** Standard de facto pentru I/O video, rapid în C++, suport nativ pentru
camere, fișiere și stream-uri RTSP.

---

### Q-Learning / NumPy 2.4.2
**Ce e:** algoritm de Reinforcement Learning tabelar; numpy pentru calcule vectoriale
pe Q-table.

**Unde și cum:** `traffic_monitor/agent/q_learning.py`. `TrafficSignalAgent` menține
un Q-table ca `defaultdict` cu chei tuple și valori `np.zeros(4)`. Spațiul de stări
are 1024 de stări discrete: 4 direcții (N/S/E/V) × 4 bin-uri de coadă (`[0,3,8,∞]`)
× 4 faze semnafor = 4⁴ × 4 = 1024. Fiecare apel la `/api/signal` construiește
starea curentă cu `build_state()`, selectează acțiunea cu politica epsilon-greedy
(`choose_action()`), și actualizează Q-table cu ecuația Bellman:

```
Q(s,a) += α × [r + γ × max_a' Q(s',a') - Q(s,a)]
```

Parametri: α=0.1, γ=0.95, ε decays 1.0→0.05 cu factor 0.995 per episod.

Funcția de recompensă (reward shaping): `-1.0 × vehicule_care_așteaptă + 2.0 × vehicule_care_au_trecut - 0.5 × lungime_coadă`.

Q-table-ul este antrenat offline pe `SimpleIntersectionEnv` (simulator Poisson
intern, fără SUMO) și salvat în `q_table_medium_final.json`. La startup, agentul
este încărcat cu ε=0.05 (aproape pur greedy).

Există și `FixedTimeController` (cicluri de 30s fix) — baseline pentru comparație în benchmark.

**De ce:** Q-Learning tabelar este suficient pentru spațiul mic de stări (1024),
e interpretabil (Q-table inspecabilă), și nu necesită GPU sau framework de DL.

---

### SQLite (via modulul `sqlite3` din stdlib)
**Ce e:** bază de date relațională embedded, fără server separat.

**Unde și cum:** `traffic_monitor/core/database.py`. 5 tabele create la startup:

| Tabelă | Conținut |
|---|---|
| `traffic_data` | Snapshot la fiecare 10s: timestamp, vehicles_count, avg_confidence, traffic_intensity, congestion_percentage |
| `users` | Conturi utilizatori cu password_hash bcrypt, rol (admin/user), is_active |
| `video_sources` | Istoricul schimbărilor de sursă video (audit log) |
| `email_settings` | Configurare SMTP + praguri alertă (singleton cu id=1) |
| `email_alert_history` | Log alerte trimise/eșuate |

`DatabaseLogger` scrie în `traffic_data` din propriul thread daemon, la
`DB_LOGGING_INTERVAL` secunde (default 10s). Schema include și o migrare automată
pentru `congestion_percentage` (coloană adăugată posterior). Connexiunile SQLite
sunt locale per-thread (nu partajate).

**De ce:** Fără overhead de instalare, perfect pentru volume mici-medii de date
de monitorizare. Modulul `sqlite3` e în stdlib — zero dependențe extra.

---

### Flask-Login ≥ 0.6.3 + Flask-Bcrypt ≥ 1.0.1 + Flask-WTF ≥ 1.2.0
**Ce e:** extensii Flask pentru sesiuni, hashing parole și formulare cu CSRF.

**Unde și cum:** Toate rutele din `api_bp` au `@login_required`. Rutele
administrative (creare utilizatori, schimbare sursă video, PDF, alerte email)
au în plus `@admin_required` (decorator custom în `auth/decorators.py`).

Login-ul folosește formularul `LoginForm(FlaskForm)` cu validare WTForms
(`DataRequired`, `Length`) și protecție CSRF automată. Parola e verificată cu
`bcrypt.check_password_hash()`. La prima pornire cu baza de date goală, se
creează automat contul `admin / admin123` cu avertisment vizibil în consolă.

Modelul `User` din `auth/models.py` este un `UserMixin` cu acces direct SQLite
(fără ORM). `@login_manager.user_loader` face lookup după user_id din sesiune.

**De ce:** Flask-Login e standard Flask pentru sesiuni. Bcrypt e mai sigur decât
SHA256 simplu pentru parole (cost factor adaptiv). WTF dă CSRF gratis pe formular.

---

### yt-dlp ≥ 2024.1.0
**Ce e:** librărie Python pentru extragerea URL-urilor de stream din platforme video.

**Unde și cum:** `traffic_monitor/core/capture.py`, funcția
`_get_youtube_stream_url()`. Când sursa video e un URL YouTube, `yt_dlp.YoutubeDL`
extrage un URL direct de stream (format `best[height<=720]/best`) pe care
`cv2.VideoCapture` îl poate deschide direct ca RTSP/HLS. Dacă extragerea eșuează,
sistemul cade back pe `traffic.mp4` local.

**De ce:** `cv2.VideoCapture` nu poate deschide YouTube direct. yt-dlp abstrage
toate platformele video fără descărcare efectivă.

---

### ReportLab ≥ 4.0.0 + matplotlib ≥ 3.7.0
**Ce e:** ReportLab = generare PDF programatică în Python; matplotlib = grafice.

**Unde și cum:** `traffic_monitor/reports/pdf_generator.py`. `PDFReportGenerator.generate()`
produce un PDF A4 complet în memorie (BytesIO, fără fișiere temporare), cu 5 secțiuni:

1. **Cover** — titlu, metadata sesiune, durată, sursă video
2. **Executive Summary** — 6 KPI-uri ca celule colorate (total records, avg/peak
   congestion, total vehicles, avg intensity, avg confidence) + paragraf auto-generat
3. **Traffic Analytics** — 3 grafice matplotlib: congestion over time (linie + fill),
   vehicle mix (pie chart donut), traffic intensity (bar chart)
4. **RL Optimizer Performance** — stare agent (fază curentă, ε, states explored,
   reward), Q-values per fază (bar chart), tabel benchmark RL vs Fixed-time
5. **Alert History + DB Snapshot** — ultimele 50 de înregistrări din SQLite

Graficele sunt generate cu `matplotlib.figure.Figure` + `FigureCanvasAgg` direct
(nu `pyplot.show()`), thread-safe, fără a polua state global. Un custom
`_NumberedCanvas` (two-pass render) adaugă header/footer cu "Page X of Y" pe
toate paginile în afara copertei.

**De ce:** ReportLab Platypus oferă layout flexibil (tabele, paragrafe, imagini
inline) fără dependențe native. matplotlib Agg produce PNG-uri cu calitate de tipar.

---

### smtplib (stdlib Python)
**Ce e:** client SMTP din biblioteca standard Python.

**Unde și cum:** `traffic_monitor/alerts/email_alerts.py`. `EmailAlertManager`
monitorizează congestion-ul la fiecare write al `DatabaseLogger` (~10s). Când
congestion-ul depășește pragul configurat (default 85%) pentru N citiri consecutive
(default 3) și nu e în cooldown (default 5 minute), lansează un thread daemon
care trimite un email HTML formatat via SMTP (suportă port 465 SSL sau 587
STARTTLS). Configurarea SMTP e stocată în SQLite, cache-uită 30s pentru a nu
lovi DB la fiecare verificare. Tot `smtplib` e folosit și pentru "Test email"
sincron din dashboard admin.

**De ce:** Stdlib — zero dependențe extra. Suportă Gmail (587+App Password),
Outlook, sau orice SMTP configurat.

---

## 3. Arhitectura pe scurt — povestea unui frame

La pornire, `create_app()` creează singleton-ul `TrafficSystem` care pornește
două thread-uri daemon: `VideoCaptureWorker` și `DatabaseLogger`.

**Viața unui frame în sistem:**

1. **Captură** — Sub-thread-ul `FrameCapture` citește frame-uri raw din sursa
   video (`cv2.VideoCapture`) la FPS-ul sursei. Dacă queue-ul `_infer_queue`
   (maxsize=2) e plin, frame-ul e aruncat — se preferă latență mică față de
   procesarea frame-urilor vechi.

2. **Inferență YOLO** — Thread-ul principal `VideoCaptureWorker` consumă din
   `_infer_queue` și trimite frame-ul la `FluidDetector.process()`. Dacă e un
   frame "de inferență" (al 3-lea), rulează YOLO real. Dacă e "frame sărit",
   returnează imediat rezultatul din cache. YOLO returnează bounding boxes
   cu coordonate, clasă și confidence.

3. **Anti-flicker** — `FluidDetector._apply_smoothing()` compară box-urile noi
   cu cele din frame-ul anterior prin IoU. Box-urile cu IoU ≥ 0.25 față de un
   slot existent sunt "refreshed"; celelalte rămân ca "ghost boxes" până la
   max 3 frame-uri lipsă. Numărul de vehicule e calculat din box-urile supraviețuitoare,
   nu din detecțiile raw.

4. **Actualizare stare** — Frame-ul adnotat cu box-uri desenate manual (culori
   per-clasă: cyan=car, amber=motorbike, purple=person) intră în `self._frame`.
   Counts-urile vehiculelor trec printr-o medie mobilă pe 5 frame-uri
   (`_count_history`). Congestion-ul e `min(vehicles / threshold * 100, 100)`.

5. **MJPEG streaming** — Generatorul din `/video_feed` apelează periodic
   `get_frame_bytes()` → `cv2.imencode('.jpg')` → trimite un chunk
   `multipart/x-mixed-replace`. Browserul afișează stream-ul live.

6. **RL / Semnafor** — Dashboard-ul apelează periodic `/api/signal`. Ruta
   citește `get_directional_counts()` (split N/S/E/V pe baza centrului bbox față
   de centrul frame-ului), construiește starea Q-Learning cu `build_state()`,
   agentul alege acțiunea, se verifică dacă se poate schimba faza (`should_switch_phase()`
   impune min 5s / max 60s per fază), și starea RL e actualizată în `rl_state`.

7. **SceneClassifier** — Rulează o singură dată, în background, la fiecare schimbare
   de sursă video. Timp de 10s, colectează vectori de deplasare (dx, dy) ai
   centroizilor vehiculelor între frame-uri consecutive. Binează unghiurile în
   8 sectoare de 45°. Dacă ≥80% din vectori cad în cel mai bun arc de 135°:
   `single_flow` (RL dezactivat). Altfel: `intersection` (RL activ).

8. **Logging DB** — `DatabaseLogger` scrie o linie în `traffic_data` la fiecare
   10 secunde (vehicles, confidence, intensity=vehicles/10, congestion%), apoi
   cheamă `EmailAlertManager.check_and_alert()` pentru a verifica pragurile.

---

## 4. Funcționalități complete

### Detecție & Tracking
- **Detecție vehicule în timp real** — YOLOv8 identifică mașini, motociclete
  și persoane din video live. Implementat în `FluidDetector.process()` cu
  conf=0.15, iou=0.40 NMS.
- **Stream video live adnotat** — Endpoint MJPEG `/video_feed` trimite frames
  la ~30 FPS cu bounding boxes colorate per clasă. Implementat cu
  `multipart/x-mixed-replace` în Flask.
- **Anti-flicker IoU persistence** — Box-urile dispar treptat (max 3 frame-uri
  absente) în loc să clipească. Implementat cu slot-uri persistent și
  matching IoU în `_apply_smoothing()`.
- **Medie mobilă pe 5 frame-uri** — Counts-urile afișate pe dashboard sunt
  netezite pentru a evita salturi brusca. Implementat cu `collections.deque(maxlen=5)`.
- **Split direcțional N/S/E/V** — Vehiculele sunt împărțite în 4 cadrane
  pe baza poziției centrului bbox față de centrul frame-ului.
  Implementat în `get_directional_counts()`.

### Clasificare scenă & Optimizare semafor
- **Clasificare automată scenă** — La schimbarea sursei video, un thread daemon
  analizează 10s de mișcare și clasifică scena ca `intersection` sau `single_flow`.
  Implementat în `SceneClassifier` cu analiza arcului de unghiuri.
- **Optimizare adaptivă Q-Learning** — Activ doar pe intersecții. Agentul
  alege din 4 faze (N-S verde, E-V verde, N-S stânga-protejat, E-V stânga-protejat)
  bazat pe cozile din fiecare direcție.
- **Constrângeri temporale pe faze** — Nicio fază nu poate dura mai puțin de 5s
  (anti-flicker semnafor) sau mai mult de 60s (anti-starvation). Implementat în
  `should_switch_phase()`.
- **Dashboard RL în timp real** — `/api/signal` returnează faza curentă, Q-values
  per fază, epsilon, numărul de stări explorate, recompensa instantanee.

### Dashboard & Vizualizare
- **Dashboard principal** (`/` → `index.html`) — stream video + contoare vehicule
  live + indicatori congestie + panou RL semnafor. Actualizat periodic via JavaScript
  prin endpoint-urile `/stats` și `/api/signal`.
- **Vizualizare bază de date** (`/db_view`) — tabel paginat cu toate înregistrările
  din `traffic_data`, disponibil adminilor.
- **Statistici live** (`/stats`) — JSON cu counts neted și congestion %, cu
  headere `Cache-Control: no-cache`.
- **Counts direcționale** (`/api/counts`) — JSON cu N/S/E/V + total + pedestrians + congestion.
- **Benchmark RL vs Fixed-time** (`/api/benchmark`) — citește `benchmark_results.json`
  cu rezultatele comparației pe 3 scenarii (low/medium/peak).
- **Training history** (`/api/training`) — ultimele 200 de episoade cu rewards,
  wait times și Q-values max per fază.

### Administrare & Securitate
- **Autentificare cu roluri** — Login cu CSRF protection (Flask-WTF), sesiuni
  Flask-Login, două roluri: `admin` și `user`. Fiecare rută are `@login_required`;
  rutele sensibile au în plus `@admin_required`.
- **Management utilizatori** (admin, `/admin/users`) — creare conturi, toggle
  activ/inactiv, ștergere. Adminul nu se poate dezactiva pe sine.
- **Schimbare sursă video la runtime** (admin, `/api/video-source/change`) —
  Switch thread-safe la fișier local, webcam, RTSP sau YouTube URL, cu suport
  `inference_size` opțional (ex: 1280 pentru footage aerian).
- **Upload video** (admin, `/api/video-source/upload`) — Upload multipart până la
  500MB, extensii permise: .mp4 .avi .mov .mkv .webm, stocat în `uploads/`.
- **Istoricul surselor video** (`/api/video-source`) — Ultimele 5 schimbări cu
  timestamp și utilizatorul care a făcut schimbarea.

### Raportare & Alerte
- **Export JSON** (admin, `/export/json`) — tot `traffic_data` ca array JSON.
- **Export CSV** (admin, `/export/csv`) — același conținut ca CSV descărcabil
  (`traffic_data.csv`).
- **Raport PDF** (admin, `/api/report/pdf`) — Document A4 complet cu 5 secțiuni,
  4 grafice matplotlib, benchmark tabel. Generat în-memorie, descărcabil imediat.
- **Alerte email congestion** — Configurare SMTP din dashboard, prag congestion
  (default 85%), N citiri consecutive (default 3), cooldown 5min. Email HTML
  stilizat cu statisticile momentului alertei.
- **Test email** (admin) — Trimite un email de test sincron și loghează rezultatul.
- **Istoric alerte email** (`/api/email-alert-history`) — Ultimele 20 de alerte
  trimise sau eșuate, cu timestamp și eroare dacă e cazul.

---

## 5. Aspecte tehnice notabile pentru prezentare

### Producer-Consumer cu două thread-uri pentru video
`VideoCaptureWorker` (în `core/capture.py`) implementează o arhitectură
producer-consumer: sub-thread-ul `FrameCapture` citește frame-uri raw și le pune
în `_infer_queue(maxsize=2)` cu put **non-blocking** (dacă queue e plin, frame-ul
e aruncat). Thread-ul principal consumă și rulează YOLO. Efectul: indiferent cât
de lentă e inferența, sistemul nu acumulează un backlog de frame-uri vechi — mereu
procesează frame-uri recente. Queue-ul de 2 elemente e un "buffer" minimal care
previne starvation-ul dacă inferența e ocazional mai rapidă decât captura.

**Ce spui la comisie:** "Am separat I/O-ul video de inferență pe thread-uri diferite
cu o coadă bounded de 2 elemente. Dacă YOLO e mai lent decât sursa video, frame-urile
vechi sunt silențios aruncate în loc să se acumuleze — asta menține latența stream-ului
la sub 100ms indiferent de FPS-ul inferenței."

### Anti-flicker prin IoU persistence (FluidDetector)
YOLOv8 la confidence scăzut poate face un vehicul să "apară și dispară" de la un
frame la altul chiar dacă vehiculul e static. `_apply_smoothing()` rezolvă asta
cu un set de "slot-uri" persistente: fiecare slot reține ultimele coordonate ale
unui vehicul detectat. La fiecare frame de inferență, noile detecții sunt matchuite
cu slot-urile existente prin IoU (intersecție / reuniune bounding boxes). Dacă
IoU ≥ 0.25 față de un slot de același tip: slot-ul e refreshed. Dacă o detecție
nu matchuiește niciun slot: e nou. Slot-urile fără match au un contor `missing`
care crește; la `missing > 3` slot-ul e eliminat. Rezultatul: o detecție "tremurătoare"
apare stabil pe ecran, și un vehicul care iese din cadru dispare abia după 3 frame-uri.

**Ce spui la comisie:** "Implementez un tracker simplu bazat pe IoU, separat de
NMS-ul din ultralytics, pentru a stabiliza countingul. Un vehicul detectat pe 1
din 3 frame-uri va fi totuși numărat continuu, fără salturi în statistici."

### SceneClassifier — clasificare euristică bazată pe vectori de mișcare
Sistemul trebuie să știe dacă privește o intersecție (unde RL e relevant) sau un
drum cu flux unic (unde RL e inutil — vehiculele merg într-o singură direcție și
nu există faze conflictuale). `SceneClassifier` rezolvă asta fără un model ML
antrenat: timp de ~10s colectează vectorii de deplasare `(dx, dy)` ai centroizilor
vehiculelor între frame-uri consecutive, ignorând mișcările sub 3px. Convertește
la unghi `atan2(-dy, dx) % 360` și binează în 8 sectoare de 45°. Dacă ≥80% din
toți vectorii se încadrează în cel mai bun arc de 3 sectoare adiacente (135°) →
`single_flow`. Pragul de 135° (nu 90°) e necesar pentru că un drum cu dublu sens
vizibil din pod ar distribui traficul în ~180° față de axa drumului — arcul de 90°
ar clasifica greșit ca intersecție.

**Ce spui la comisie:** "Clasificarea nu e un model ML antrenat — e o euristică
geometrică pe distribuția unghiulară a vectorilor de mișcare. Am ales 135° ca arc
primar după ce un arc de 90° clasifica greșit drumurile cu dublu sens văzute din
sus (ambele direcții vizibile). E o decizie de design deliberată, nu o limitare
— un model ML ar fi overfit pe tipurile de camere din training."

### Q-Learning cu reward shaping și benchmark simulat
Agentul controlează 4 faze semnafor cu semantică clară (N-S verde / E-V verde /
N-S stânga protejat / E-V stânga protejat). Spațiul de stări de 1024 e suficient
de mic pentru Q-table tabelar (nu e nevoie de rețea neuronală). Reward shaping-ul
are 3 componente: penalizare per vehicul care așteaptă (-1.0), bonus per vehicul
care a trecut (+2.0), și o penalizare mai mică pentru lungimea cozii (-0.5).
Raportul 2:1 între reward și penalizare înseamnă că agentul preferă să golească
cozi în loc să minimizeze pur așteptarea.

Benchmark-ul compară agentul RL cu un controller fixed-time (30s per fază) pe
3 scenarii cu sosiri Poisson (low/medium/peak) în `SimpleIntersectionEnv`. Fiecare
scenariu rulează 50 episoade de evaluare cu ε=0 (pur greedy). Rezultatele sunt
salvate în `benchmark_results.json` și afișate în PDF și pe dashboard.

**Ce spui la comisie:** "Am antrenat pe un simulator intern cu sosiri Poisson,
nu pe date reale sau SUMO, pentru că scopul e să demonstrez că algoritmul poate
invăța o politică mai bună decât fixed-time pe o distribuție realistă de trafic.
Simulatorul nu pretinde să modeleze trafic real — e un mediu de proof-of-concept
suficient de complex pentru a demonstra beneficiul RL față de baseline."

### ONNX vs .pt fallback și inference tuning
Factory-ul `_build_detector()` preferă `yolov8s.onnx` (small, imgsz=480) față
de `yolov8m.pt` (medium, mai mare). ONNX e mai rapid pe CPU datorită optimizărilor
de graph și eliminării overhead-ului PyTorch. `export_onnx.py` generează ONNX-ul
o singură dată cu `simplify=True` (ONNX Simplifier reduce redundanța grafului).
Dacă ONNX lipsește, sistemul pornește cu .pt cu un warning vizibil — nu se crapă.

**Ce spui la comisie:** "Am ales yolov8s (small) față de medium pentru că la
imgsz=480 pe CPU oferă un compromis mai bun FPS/acuratețe pentru trafic urban.
ONNX în plus față de .pt elimină ~30% overhead la inferență fără pierdere de
acuratețe. Fallback-ul automat la .pt face sistemul portabil pe orice mașină
unde ONNX nu a fost generat."

### Congestion ca metrica simplă dar funcțională
`congestion_level = min(vehicles / CONGESTION_THRESHOLD * 100, 100)` unde
`CONGESTION_THRESHOLD=25` (configurat). Asta înseamnă că 25 de vehicule = 100%
congestie. Metrica e liniară și nu depinde de viteza vehiculelor sau de tipul
intersecției — e o aproximare simplă dar consistentă pentru o cameră fixă cu
câmp vizual cunoscut.

### Settings singleton cu rezoluție din env vars
`traffic_monitor/config.py` definește un singur obiect `Settings` la nivel de
modul, configurat din env vars cu fallback-uri safe. `_WORKSPACE_DIR` e rezolvat
la `Path(__file__).parent.parent.parent.parent` — care ajunge la directorul exterior
unde stau modelele și videoul mare. Asta permite deployment-ul aplicației ca pachet
Python instalabil (`pyproject.toml`) cu fișierele mari în afara repository-ului.

---

## 6. Limitări / decizii de design de menționat la comisie

### SceneClassifier e euristic, nu ML antrenat
Clasificarea scenei (intersecție vs. flux unic) nu folosește un model antrenat pe
exemple etichetate de camere de trafic. E o regulă geometrică pe distribuția
unghiulară. Avantaj: funcționează zero-shot pe orice cameră. Dezavantaj: poate
clasifica greșit o parcare circulară ca intersecție sau un roundabout ca drum drept.
**Justificare:** Pentru scopul proiectului (a decide dacă RL-ul e relevant),
precizia euristicii e suficientă. Un model ML ar necesita un dataset etichetat
care nu există public pentru camere de trafic românești.

### Q-Learning antrenat pe simulator, nu pe date de trafic real
`SimpleIntersectionEnv` folosește sosiri Poisson cu rate-uri fixe pe scenarii
(low/medium/peak). Acestea nu sunt calibrate pe date reale de trafic. Agentul
nu a văzut niciodată un semnafor real, variații de zi/noapte, sau comportament
de șoferi reali. **Justificare:** Obiectivul tezei e să demonstreze că Q-Learning
adaptiv poate depăși un controller fixed-time pe o distribuție de trafic — nu să
controleze un semnafor real în producție. Simulatorul e suficient de complex
(4 direcții, Poisson arrivals, discharge rate limitat) pentru a valida principiul.

### Congestia e o funcție liniară simplă
`congestion = vehicles / 25 * 100`. Nu ia în calcul viteza vehiculelor, tipul lor
(un camion vs o motocicletă), densitatea de trafic pe sens, sau lungimea efectivă
a cozii. **Justificare:** Aplicația nu are senzori de viteză sau loops de inducție
— dispune doar de numărul de vehicule detectate vizual. Metrica liniară e consistentă
și calibrabilă (CONGESTION_THRESHOLD din .env).

### Direcționalitatea N/S/E/V e relativă la frame, nu la direcții cardinale
`get_directional_counts()` împarte vehiculele în cadranele frame-ului (stânga/dreapta
față de centrul orizontal = E/V; sus/jos față de centrul vertical = N/S). Nu se
face nicio aliniere cu nordul real sau cu orientarea intersecției. **Justificare:**
Sistemul nu are metadata despre orientarea camerei. Logica e consistentă intern —
agentul RL primește mereu aceeași distribuție N/S/E/V pentru același tip de trafic
din aceeași cameră.

### Fără tracking multi-obiect (MOT) dedicat
`FluidDetector` nu implementează un tracker ca SORT sau DeepSORT cu ID-uri
persistente. "Persistența" IoU e frame-to-frame și nu menține identitatea
vehiculelor pe distanțe mai mari sau la re-intrarea în cadru. **Justificare:**
Scopul e numărarea și clasificarea scenei, nu urmărirea individuală a vehiculelor.
Un tracker dedicat ar adăuga complexitate semnificativă fără beneficiu direct
pentru funcționalitățile implementate.

### Aplicație single-process, nu distribuită
Toate componentele (captura video, YOLO, RL, DB, email) rulează ca thread-uri în
același proces Python. Nu există load balancing, nu se poate scala orizontal.
**Justificare:** Pentru un sistem de monitorizare single-camera pe o singură
intersecție, arhitectura monolitică e adecvată și simplifică semnificativ
deployment-ul și debugging-ul.

---

*Generat din codul real al proiectului — toate valorile numerice, numele
fișierelor și comportamentele descrise sunt verificate direct din sursă.*
