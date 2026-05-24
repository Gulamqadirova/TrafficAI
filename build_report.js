// build_report.js — TPBI Unit 12 report (person + car detection platform)
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, Footer, AlignmentType, LevelFormat, HeadingLevel,
  BorderStyle, WidthType, ShadingType, PageNumber, PageBreak, TableOfContents,
  VerticalAlign,
} = require("docx");

const OUT = "/home/claude/traffic_platform/outputs";
const CW = 9360;

const P = (t, o = {}) => new Paragraph({ spacing: { after: o.after ?? 120, line: 276 },
  alignment: o.align, children: [new TextRun({ text: t, bold: o.bold, italics: o.italic, size: o.size, color: o.color })] });
const lead = (b, r) => new Paragraph({ spacing: { after: 120, line: 276 },
  children: [new TextRun({ text: b, bold: true }), new TextRun({ text: r })] });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bullet = (t) => new Paragraph({ numbering: { reference: "bul", level: 0 },
  spacing: { after: 60, line: 276 }, children: [new TextRun(t)] });
const pb = () => new Paragraph({ children: [new PageBreak()] });
const spacer = () => P("", { after: 80 });

function figure(file, w, h, cap) {
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(`${OUT}/${file}`),
        transformation: { width: w, height: h },
        altText: { title: cap, description: cap, name: cap } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: cap, italics: true, size: 18, color: "595959" })] }),
  ];
}
const bd = { style: BorderStyle.SINGLE, size: 1, color: "BFBFBF" };
const borders = { top: bd, bottom: bd, left: bd, right: bd };
function cell(t, w, { head = false, bold = false, fill } = {}) {
  return new TableCell({ borders, width: { size: w, type: WidthType.DXA },
    shading: fill ? { fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 }, verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ children: [new TextRun({ text: t, bold: head || bold,
      color: head ? "FFFFFF" : undefined, size: 19 })] })] });
}
function table(headers, rows, widths) {
  const hr = new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, widths[i], { head: true, fill: "2E75B6" })) });
  const br = rows.map((r, ri) => new TableRow({ children: r.map((c, i) => cell(String(c), widths[i], { fill: ri % 2 ? "EAF1F8" : undefined })) }));
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: widths, rows: [hr, ...br] });
}

const children = [];

// ===== TITLE =====
children.push(
  new Paragraph({ spacing: { before: 1800, after: 120 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Unit 12: Business Intelligence", bold: true, size: 28, color: "2E75B6" })] }),
  new Paragraph({ spacing: { after: 600 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Higher National Assignment", size: 22, color: "595959" })] }),
  new Paragraph({ spacing: { after: 120 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "AI-Driven Business Intelligence Platform for Smart Surveillance and Human Activity Analytics", bold: true, size: 36 })] }),
  new Paragraph({ spacing: { before: 200, after: 1200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "A Person & Vehicle Detection, Tracking, Counting and Analytics System (the TPBI Platform)", italics: true, size: 22, color: "595959" })] }),
);
children.push(new Table({ width: { size: 6200, type: WidthType.DXA }, alignment: AlignmentType.CENTER, columnWidths: [2700, 3500],
  rows: [
    ["Role", "Junior Business Intelligence & AI Consultant"],
    ["Client", "Smart City Innovation Consortium"],
    ["Detected classes", "Person and Car (vehicle)"],
    ["Learning aims covered", "LO1, LO2, LO3, LO4"],
    ["Deliverable", "Written analysis (Tasks 1\u20134) + working code + docs"],
  ].map((r, i) => new TableRow({ children: [cell(r[0], 2700, { bold: true, fill: "EAF1F8" }), cell(r[1], 3500, { fill: i % 2 ? "FFFFFF" : "F7F9FC" })] })) }));
children.push(pb());

// ===== TOC =====
children.push(H1("Table of Contents"));
children.push(new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }));
children.push(pb());

// ===== INTRO =====
children.push(H1("Introduction"));
children.push(P("This report supports an AI-Powered Traffic and Pedestrian Business Intelligence (TPBI) platform for a smart-city innovation consortium spanning universities, municipal authorities, transport departments and private security organisations. The platform uses computer vision and artificial intelligence to detect, track and count two classes of object \u2014 people and cars \u2014 in live video, and turns those observations into business intelligence: dashboards, KPIs, anomaly alerts, demand forecasts and a natural-language assistant. It is designed for deployment at public streets, smart campuses, transport hubs and commercial areas."));
children.push(P("The report follows the four assignment tasks. Task 1 explains the AI and BI building blocks; Task 2 describes the data pipeline and architecture; Task 3 applies the analytics and presents the dashboards, KPIs and reports; and Task 4 critically evaluates the system and recommends improvements. The discussion is grounded in a complete, working reference implementation (ten Python modules orchestrated by pipeline.py)."));
children.push(lead("A note on the figures: ", "every chart, KPI and statistic was produced by running the implementation end-to-end. In the demonstration run the system processed a 150-frame two-class traffic clip, aggregated a full day of four-camera data (1,152 records), detected 24 anomalous readings (including an evening traffic-congestion event) and generated a four-sheet Excel BI report \u2014 in under a minute on commodity hardware."));
children.push(pb());

// ===== TASK 1 =====
children.push(H1("Task 1 \u2013 AI and BI Overview"));
children.push(P("This task explains the technologies that make the platform an intelligent decision-support system: AI monitoring systems, the YOLO and OpenCV computer-vision stack, dashboard reporting, the AI chatbot, and the benefits of smart surveillance."));

children.push(H2("1.1 AI monitoring systems"));
children.push(P("An AI monitoring system continuously interprets live sensor data rather than merely recording it. Here, each video frame is analysed in a loop: a detection model locates every person and car, a tracker follows each object across frames to build trajectories, and counts are compared with learned norms so abnormal conditions raise alerts automatically. This converts a passive camera into an active source of structured, decision-ready signals \u2014 for example, detecting a developing traffic jam the moment road occupancy crosses a threshold, instead of relying on an operator to notice it."));
children.push(lead("In the implementation: ", "the monitoring loop is analyse_video(), which reads each frame, detects both classes, updates per-class trackers and directional line counters, and emits per-frame counts and confidence scores that feed every downstream analytic."));

children.push(H2("1.2 YOLO and OpenCV usage"));
children.push(P("OpenCV (Open Source Computer Vision Library) provides the video-analytics foundation: decoding the stream, reading and writing frames, colour conversion, drawing annotations (class-coloured bounding boxes, identifiers, the counting line and the heads-up display), and writing the annotated output video. It is fast, mature and edge-friendly."));
children.push(P("YOLO (You Only Look Once) is the deep-learning object detector that performs the recognition. As a single-stage detector it predicts all bounding boxes and class probabilities in one pass, enabling real-time operation. In this platform YOLOv8 detects both the \u2018person\u2019 and \u2018car\u2019 COCO classes, returning a box, class and confidence for each object; OpenCV then consumes those boxes for tracking, counting and visualisation."));
children.push(lead("Engineering decision \u2013 graceful degradation: ", "the MultiClassDetector prefers YOLOv8 but falls back to OpenCV\u2019s built-in HOG pedestrian detector (people) and a Haar/shape-based vehicle detector (cars) when YOLO weights are unavailable. This resilience pattern keeps a node producing two-class data even without a model download, and is why the demonstration still detected both classes (people at 0.68 and cars at 0.60 mean confidence)."));

children.push(H2("1.3 Dashboard reporting"));
children.push(P("Dashboard reporting turns analytics into something a decision-maker reads at a glance: a few headline KPIs (peak road occupancy, total cars, total people, anomaly count, busiest site) with drill-down into trends, comparisons and exceptions. The reporting layer (bi_dashboard.py) computes the KPIs and renders a car-versus-pedestrian flow chart, a road-occupancy heatmap, a traffic-anomaly timeline, a site-comparison bar chart and a forecast overlay, then publishes a four-sheet Excel workbook with a native chart \u2014 mirroring a Power BI deployment."));

children.push(H2("1.4 AI chatbot functions"));
children.push(P("The AI chatbot is a natural-language interface so non-technical staff can interrogate the analytics without SQL or a BI tool. A traffic controller can ask \u2018was there any congestion today?\u2019 or \u2018is the main road safe?\u2019 and get a plain-English answer drawn from the live KPIs. The reference TrafficChatbot is a transparent, auditable intent engine that runs offline; in production it would be backed by a large-language-model API with retrieval over the same tables. Sample exchanges from the run:"));
children.push(table(["User question", "Chatbot response (verbatim from run)"],
  [
    ["Was there any congestion today?", "24 anomalous readings detected; most severe at JCT_MainRoad at 17:40 with 83 cars (200% of road capacity)\u2026"],
    ["What is the busiest zone?", "Peak road occupancy at JCT_MainRoad at 17:30 (200% \u2014 Congested); busiest car site Commercial Area\u2026"],
    ["Is the main road safe?", "Congestion risk detected \u2014 recommend signal-timing review and operator alert."],
  ], [3200, 6160]));
children.push(spacer());

children.push(H2("1.5 Benefits of smart surveillance"));
children.push(P("Smart surveillance of people and vehicles delivers value across safety and operations:"));
children.push(bullet("Traffic and crowd management \u2013 early detection of congestion and overcrowding enables intervention (signal retiming, opening lanes, staff dispatch) before gridlock or a safety incident, as shown by the platform flagging the 17:30 congestion event in real time."));
children.push(bullet("Operational efficiency \u2013 directional car and pedestrian counts inform signal timing, parking management, staffing and retail/transport planning."));
children.push(bullet("Proactive, predictive decisions \u2013 next-hour forecasts let operators pre-empt an expected peak rather than react to it."));
children.push(bullet("Objective, auditable evidence \u2013 each alert carries its triggering reason (which detector fired, which z-score), supporting accountable decisions."));
children.push(bullet("Cost reduction \u2013 automation focuses scarce human attention on genuine exceptions instead of watching every screen."));
children.push(P("These must be balanced against privacy, accuracy and governance, addressed in Task 4."));
children.push(pb());

// ===== TASK 2 =====
children.push(H1("Task 2 \u2013 Data Pipeline and System Architecture"));
children.push(P("This task describes the end-to-end pipeline \u2014 collection, storage, processing, real-time analytics and the dashboard system \u2014 and the architecture that makes it scalable, integrable and sustainable. The named tools (Python, YOLO, OpenCV, Power BI, SQL) are positioned within it, and the storage and modelling choices for structured, semi-structured and unstructured data are justified."));

children.push(H2("2.1 Architecture overview"));
children.push(P("The platform follows a layered, \u2018medallion\u2019-style flow: Ingest \u2192 Perceive \u2192 Analyse \u2192 Present \u2192 Assist. Each layer has one clear responsibility and a stable interface, so any component can be upgraded independently \u2014 a deliberate strategy for sustainability and integration."));
children.push(table(["Layer", "Responsibility", "Primary tools"],
  [
    ["Ingest", "Capture video; aggregate to a 2-class time-series", "RTSP/Kafka, Python, OpenCV"],
    ["Perceive", "Detect, track and count people & cars per frame", "YOLOv8, OpenCV, Python"],
    ["Analyse", "Anomaly detection and demand forecasting", "Python, scikit-learn"],
    ["Present", "KPIs, charts, dashboards, reports", "Power BI, openpyxl, SQL"],
    ["Assist", "Natural-language querying", "LLM API / chatbot"],
  ], [1500, 4960, 2900]));
children.push(spacer());

children.push(H2("2.2 Data collection"));
children.push(P("Data originates as continuous video from cameras at each site. At the edge, each node decodes its stream with OpenCV and runs detection locally, so only lightweight structured records (timestamp, zone, car-count, person-count, confidence, crossing events) travel across the network \u2014 not raw video. This edge-first strategy cuts bandwidth, protects privacy and improves resilience; in a full deployment these records are published to a streaming bus such as Apache Kafka. In the implementation, collection is simulated by synthetic_data.py, which produces a real OpenCV traffic clip and a realistic full-day, four-camera, two-class dataset (1,152 records)."));

children.push(H2("2.3 Data storage"));
children.push(P("Surveillance generates three data shapes; differentiating them is essential to cost and performance."));
children.push(table(["Data type", "Examples here", "Suitable storage"],
  [
    ["Structured", "Car/person counts per interval, KPIs, zone metadata", "Relational DB / SQL warehouse"],
    ["Semi-structured", "Detection & crossing events, model/config (JSON)", "Document store / data lake (JSON, Parquet)"],
    ["Unstructured", "Raw and annotated video frames", "Object / blob storage (S3-style)"],
  ], [2000, 4360, 3000]));
children.push(spacer());
children.push(lead("Storage models compared. ", "A relational/SQL model suits the structured counts and KPIs the dashboards query (schema, fast aggregation, native Power BI integration). A data lake suits schema-flexible, append-heavy capture and cheap retention of raw video. A modern deployment uses a lakehouse hybrid: a lake for raw/intermediate layers and a SQL serving layer for BI."));
children.push(lead("Data modelling. ", "For the serving layer a dimensional (star) model is most suitable: a fact table of observations (grain = one zone per interval, measures car_count and person_count) with dimensions for time, zone and site. This read-optimised shape gives Power BI fast aggregation across site \u2192 zone and across time-of-day \u2014 the exact slice-and-dice the dashboards perform \u2014 whereas a fully normalised model would slow analytical queries."));

children.push(H2("2.4 Data processing"));
children.push(P("The platform combines real-time and batch processing because they answer different questions."));
children.push(bullet("Real-time / stream processing handles per-frame detection, tracking and threshold alerting that must occur within seconds for safety (e.g. a congestion alert). Latency is the priority."));
children.push(bullet("Batch processing handles model (re)training, historical aggregation, the daily anomaly scan and forecasting, where accuracy and full history matter more than immediacy."));
children.push(P("This is the classic speed-layer-versus-batch-layer (Lambda-style) trade-off: real-time gives immediate, approximate awareness; batch gives accurate, comprehensive analytics. Using both is justified because no single mode satisfies both the sub-second safety requirement and the high-accuracy analytical requirement."));

children.push(H2("2.5 Real-time analytics"));
children.push(P("Real-time analytics computes live road and pedestrian occupancy per zone as frames arrive, classifies each reading into a congestion band (Free-flow / Busy / Heavy / Congested) against capacity, and fires an alert when the Congested band or the 85% capacity line is crossed. Because detection runs at the edge and only compact records traverse the network, this keeps pace with the stream across many cameras."));

children.push(H2("2.6 Dashboard system"));
children.push(P("The serving layer reads the structured warehouse and presents it through dashboards. In production this is Power BI: a scheduled or streaming refresh pulls from the SQL warehouse, and pages expose KPI cards, the road-occupancy heatmap, flow trends and the forecast, with row-level security per stakeholder. The reference implementation reproduces this with matplotlib charts and an openpyxl Excel workbook (KPI Dashboard, Hourly Flow with a native chart, Anomalies, Forecast)."));
children.push(lead("Tooling summary. ", "Python orchestrates and analyses; YOLO detects; OpenCV handles video I/O and analytics; SQL is the warehouse/query layer for the structured serving model; Power BI is the enterprise dashboarding tool. Each is used where strongest, with narrow interfaces so components can be swapped without disruption."));
children.push(pb());

// ===== TASK 3 =====
children.push(H1("Task 3 \u2013 AI and Analytics Application"));
children.push(P("This task applies the analytics \u2014 human and vehicle detection, counting, object tracking, anomaly detection and predictive analytics \u2014 and presents the resulting charts, KPIs and reports. All outputs were generated by running the implementation end-to-end."));

children.push(H2("3.1 Human and vehicle detection"));
children.push(P("Each frame is passed to the detector, which returns a class-labelled bounding box and confidence for every person and car. In the run the system processed 150 frames and detected both classes, peaking at four people and six cars, with mean confidences of 0.68 (people) and 0.60 (cars). The annotated frame below shows people in green and cars in orange, with class labels, confidence, track identifiers and the red counting line."));
children.push(...figure("detection_frame.png", 480, 270, "Figure 1 \u2013 Multi-class detection: people (green) and cars (orange) with labels, confidence and track IDs."));

children.push(H2("3.2 Counting (cars and pedestrians)"));
children.push(P("Per-frame detections are aggregated into counts per zone for each class, expressed as occupancy against capacity and classified into congestion bands. Across the day the system estimated 12,175 car detections and 36,140 pedestrian detections; the Commercial Area was the busiest for cars and the Transport Hub the busiest for people. The flow chart contrasts the sharp twin commuter peaks of car traffic with the broader daytime pedestrian curve at the busiest junction."));
children.push(...figure("chart_flow_trends.png", 560, 245, "Figure 2 \u2013 Car vs pedestrian flow at the main junction (note the twin car rush-hours)."));
children.push(...figure("chart_road_heatmap.png", 560, 210, "Figure 3 \u2013 Mean road occupancy by zone and hour."));

children.push(H2("3.3 Object tracking"));
children.push(P("A per-class centroid tracker assigns a persistent identifier to each object across frames, converting stateless detections into trajectories. Tracking the two classes independently prevents identity confusion between a pedestrian and a vehicle, and enables counting unique objects (not re-counting each frame) and directional line-crossing counts (e.g. cars eastbound versus westbound) \u2014 the core analytic of vehicle-counting systems. The tracker is pure-NumPy for edge deployment; a Kalman/Deep-SORT tracker would replace it for very dense scenes behind the same interface."));

children.push(H2("3.4 Anomaly detection"));
children.push(P("Two complementary techniques are combined. An Isolation Forest (unsupervised) learns the normal joint pattern of car and pedestrian flow across the day and flags multivariate outliers without labels. A per-zone, per-metric z-score gives a transparent statistical backstop. A reading is flagged if either fires, and each alert records its reason for auditability. In the run, 24 anomalous readings were detected; the most severe was at the main road at 17:40 with 83 cars (200% of road capacity, the Congested band), flagged by both methods (car z-score 4.1)."));
children.push(...figure("chart_anomaly_timeline.png", 560, 238, "Figure 4 \u2013 Traffic anomaly timeline at the main road; red points are flagged anomalies, the dashed line is road capacity."));

children.push(H2("3.5 Predictive analytics"));
children.push(P("Forecasting makes the platform proactive. For each zone, interpretable Ridge models on cyclical time-of-day features project the next hour of car and pedestrian flow, so operators are warned before a road fills. Treating 08:00 as \u2018now\u2019, the model projected the morning car flow at the main road and flagged whether the 85% road-capacity alert line would be approached. The model is intentionally simple and cheap to retrain at the edge; production would add richer models (gradient boosting, Prophet, LSTM) and external signals such as events and weather."));
children.push(...figure("chart_forecast.png", 560, 238, "Figure 5 \u2013 Next-hour traffic forecast for the main road with the 85% road-capacity alert line."));

children.push(H2("3.6 KPIs, reports and dashboards"));
children.push(P("The headline KPIs from the run are summarised below, with the site-comparison chart. These are published to the four-sheet Excel BI report (and would refresh a Power BI dataset in production)."));
children.push(table(["KPI", "Value"],
  [
    ["Total car detections", "12,175"],
    ["Total pedestrian detections", "36,140"],
    ["Peak road occupancy", "200% \u2014 JCT_MainRoad at 17:30 (Congested)"],
    ["Anomalies detected", "24"],
    ["Busiest car / pedestrian site", "Commercial Area / Transport Hub"],
    ["Zones / sites monitored", "4 / 4"],
    ["CV backend / frames", "opencv_fallback / 150"],
  ], [4560, 4800]));
children.push(spacer());
children.push(...figure("chart_site_comparison.png", 545, 240, "Figure 6 \u2013 Daily volume by site: cars versus people."));

children.push(H2("3.7 Justification, strengths and limitations"));
children.push(P("The workflow is justified against stakeholders and objectives: safety teams need low-latency detection and congestion alerts; operations teams need the flow and occupancy analytics for signal/parking planning; managers need forecasts and the chatbot for proactive, accessible decisions. Unsupervised anomaly detection reflects that incidents are rare and unlabelled, and the dimensional serving model reflects the need for fast dashboard aggregation."));
children.push(table(["Technique", "Strengths", "Limitations"],
  [
    ["YOLO / OpenCV detection", "Real-time; multi-class; edge-deployable", "Accuracy falls in dense traffic, occlusion, poor light"],
    ["Per-class centroid tracking", "Lightweight; unique counts and direction", "ID switches when paths cross or detections drop"],
    ["Isolation Forest + z-score", "Unsupervised; explainable; no labels", "May flag legitimate rare peaks; needs threshold tuning"],
    ["Ridge forecasting", "Fast; interpretable; cheap to retrain", "Underfits sharp surges; ignores external drivers"],
  ], [2300, 3530, 3530]));
children.push(pb());

// ===== TASK 4 =====
children.push(H1("Task 4 \u2013 System Evaluation and Recommendations"));
children.push(P("This task evaluates the platform against system performance, AI accuracy, scalability, data quality and privacy/security, assesses it against the four \u2018V\u2019s of data, interprets the governance context, and recommends improvements."));

children.push(H2("4.1 System performance"));
children.push(P("End-to-end the pipeline completed in under a minute in the demonstration, including two-class video processing, a full-day anomaly scan, forecasting for four zones and generation of all charts and the Excel report. The edge-first design \u2014 detecting locally and transmitting only compact records \u2014 keeps central analytics responsive as cameras are added. The principal real-time constraint is detection throughput per node; YOLOv8-nano is chosen precisely because it sustains real-time frame rates on modest hardware, and the OpenCV fallback guarantees continued operation if a node cannot load it."));

children.push(H2("4.2 AI accuracy"));
children.push(P("Detection accuracy was solid for the controlled scene (both classes detected; 0.68 person and 0.60 car mean confidence), but real-world accuracy degrades with occlusion, dense traffic, low light, weather and camera angle, and the OpenCV fallback is weaker than YOLO. Tracking accuracy is bounded by ID-switching when trajectories cross, which is more frequent for fast vehicles. The forecaster\u2019s honest validation error (MAE) shows the simple model underfits sharp surges. Accuracy should be monitored continuously with periodic ground-truth audits and models retrained on site-specific data rather than assumed constant."));

children.push(H2("4.3 Scalability and integration"));
children.push(P("Scalability is architectural. Detection is horizontally scalable \u2014 each camera is an independent edge node, so adding cameras adds compute linearly. A streaming bus decouples producers from consumers and absorbs spikes; lakehouse storage scales independently for raw video and structured serving; and narrow interfaces let any component (the forecaster, the detector, the YOLO weights) be upgraded without disruption. Integration is eased by standard protocols (RTSP for cameras, SQL for the warehouse, Power BI for reporting) and by keeping per-site configuration in one file so the same code redeploys to a street, junction, campus gate or car-park. The main long-term risks are model drift and storage growth, addressed below."));

children.push(H2("4.4 Data quality \u2013 the four V\u2019s"));
children.push(P("Pipeline effectiveness is assessed against volume, variety, velocity and veracity:"));
children.push(bullet("Volume \u2013 continuous multi-camera video is high-volume; the edge-first design and lakehouse storage keep this tractable by never centralising raw video unnecessarily."));
children.push(bullet("Variety \u2013 the system handles structured (counts, KPIs), semi-structured (event/config JSON) and unstructured (video) data, each routed to a fit-for-purpose store."));
children.push(bullet("Velocity \u2013 the speed layer meets the sub-second alerting requirement while the batch layer handles high-accuracy historical analytics."));
children.push(bullet("Veracity \u2013 the weakest V: detections are probabilistic and noisy, so the platform attaches confidence scores, combines two anomaly methods and reports honest forecast error; camera-health and dropped-frame checks are recommended to protect downstream BI."));

children.push(H2("4.5 Privacy, security and governance"));
children.push(P("AI surveillance of people and vehicles in public space carries significant legal and ethical obligations, particularly under data-protection law (in the UK/EU, the GDPR and the Data Protection Act); vehicle number plates and identifiable individuals are personal data. Key requirements include a lawful basis and a documented Data Protection Impact Assessment; data minimisation and purpose limitation (processing counts and trajectories, not identities, with the edge-first design so faces and plates need not be stored); proportionality and clear public signage; strict access control, encryption in transit and at rest, and audit logging; defined retention and deletion schedules; and human oversight so consequential decisions are not fully automated. The platform\u2019s explainable alerts directly support the accountability that governance frameworks demand. Ethically, the consortium should guard against function creep (re-using a traffic-safety system for unrelated tracking) and against detection bias across vehicle types or pedestrian demographics."));

children.push(H2("4.6 Recommendations"));
children.push(bullet("Deploy production YOLOv8 weights per edge node and retain the OpenCV fallback for resilience; benchmark accuracy per site before go-live."));
children.push(bullet("Add continuous data-quality monitoring (camera health, dropped frames, confidence drift) so BI is not corrupted by silent sensor failures."));
children.push(bullet("Upgrade the forecaster to capture sharp surges (gradient boosting / Prophet / LSTM) and add exogenous signals (events, weather, timetables)."));
children.push(bullet("Introduce automatic model-drift detection and scheduled retraining for long-term sustainability."));
children.push(bullet("Formalise governance: complete a DPIA, define retention/deletion, enforce row-level security in Power BI, and keep humans in the loop for consequential actions."));
children.push(bullet("Replace the rule-based chatbot with an LLM-backed assistant using retrieval over the analytics tables, with guardrails and answer logging."));

children.push(H2("Conclusion"));
children.push(P("The TPBI platform demonstrates a complete, working AI-driven business-intelligence pipeline that detects, tracks and counts both people and cars in video, finds anomalies, forecasts demand, and serves KPIs, dashboards, reports and a natural-language assistant. The layered, edge-first architecture is scalable, integrable and sustainable, and the demonstration produced concrete, decision-relevant results \u2014 most notably the real-time detection of an evening traffic-congestion event. Realising the platform at scale depends less on the algorithms than on disciplined data quality, continuous accuracy monitoring and robust privacy governance, which the recommendations prioritise."));
children.push(pb());

children.push(H1("References"));
[
  "Bochkovskiy, A., Wang, C.-Y. and Liao, H.-Y. M. (2020) YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv:2004.10934.",
  "Bradski, G. (2000) \u2018The OpenCV Library\u2019, Dr. Dobb\u2019s Journal of Software Tools.",
  "Kimball, R. and Ross, M. (2013) The Data Warehouse Toolkit. 3rd edn. Wiley.",
  "Liu, F. T., Ting, K. M. and Zhou, Z.-H. (2008) \u2018Isolation Forest\u2019, IEEE ICDM, pp. 413\u2013422.",
  "Marz, N. and Warren, J. (2015) Big Data: Principles and Best Practices of Scalable Real-Time Data Systems. Manning.",
  "Information Commissioner\u2019s Office (2023) Guidance on AI and Data Protection. ICO.",
  "Ultralytics (2023) YOLOv8 Documentation. Available at: https://docs.ultralytics.com.",
].forEach(r => children.push(new Paragraph({ spacing: { after: 100, line: 276 },
  indent: { left: 360, hanging: 360 }, children: [new TextRun({ text: r, size: 20 })] })));

const doc = new Document({
  styles: { default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Arial", color: "1F4E79" }, paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: "Arial", color: "2E75B6" }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
    ] },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 600, hanging: 300 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF", space: 4 } },
      children: [new TextRun({ text: "Unit 12: Business Intelligence \u2014 Person & Vehicle Analytics Platform", size: 16, color: "808080" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Page ", size: 16, color: "808080" }),
        new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "808080" }),
        new TextRun({ text: " of ", size: 16, color: "808080" }),
        new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 16, color: "808080" })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => { fs.writeFileSync(`${OUT}/TPBI_Report.docx`, buf);
  console.log("Report written:", `${OUT}/TPBI_Report.docx`, buf.length, "bytes"); });
