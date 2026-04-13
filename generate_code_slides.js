const PptxGenJS = require("pptxgenjs");

const pres = new PptxGenJS();
pres.layout = "LAYOUT_16x9";
pres.author = "Michael Romano";
pres.title = "UCSF PCNSL Dataset - Code & Skill Slides";

// ── Design System (extracted from existing presentation XML) ──
const C = {
  navy: "052049",
  teal: "18837E",
  gold: "FDB515",
  ltBg: "F4F6F9",
  white: "FFFFFF",
  body: "3A4A5C",
  footer: "AABBCC",
  codeBg: "1E2028",
  codeText: "E0E0E0",
  codeKw: "81A1C1",
  codeStr: "A3BE8C",
  codeFn: "88C0D0",
  codeComment: "616E88",
  orange: "D08770",
};

const FOOTER_TEXT = "UCSF PCNSL Dataset  | Ci2 Lecture";

// ── Helper: add standard content-slide chrome ──
function addContentChrome(slide, title) {
  slide.background = { color: C.ltBg };
  // Top teal bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.08, fill: { color: C.teal },
  });
  // Title
  slide.addText(title, {
    x: 0.57, y: 0.24, w: 8.4, h: 0.57,
    fontSize: 26, fontFace: "Georgia", bold: true, color: C.navy,
    margin: 0,
  });
  // Bottom navy bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 4.88, w: 10, h: 0.45, fill: { color: C.navy },
  });
  // Footer text
  slide.addText(FOOTER_TEXT, {
    x: 0.38, y: 4.91, w: 8.76, h: 0.38,
    fontSize: 9, fontFace: "Calibri", italic: true, color: C.footer,
    margin: 0,
  });
}

// Helper: make shadow config (fresh each call to avoid mutation bug)
const mkShadow = () => ({
  type: "outer", blur: 3, offset: 1, angle: 225, color: "000000", opacity: 0.06,
});

// ══════════════════════════════════════════════════════════════
// SLIDE N1: Section Divider — "Using the Dataset"
// ══════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  slide.background = { color: C.navy };
  // Gold accent bar (vertically centered)
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.76, y: 1.95, w: 0.11, h: 1.9, fill: { color: C.gold },
  });
  // Title (vertically centered)
  slide.addText("Using the Dataset", {
    x: 1.14, y: 1.86, w: 7.6, h: 1.14,
    fontSize: 36, fontFace: "Georgia", bold: true, color: C.white,
    margin: 0, valign: "middle",
  });
  // Subtitle
  slide.addText("Code architecture, API & AI-assisted analysis", {
    x: 1.14, y: 3.04, w: 7.6, h: 0.57,
    fontSize: 18, fontFace: "Calibri", color: C.gold,
    margin: 0, valign: "middle",
  });
}

// ══════════════════════════════════════════════════════════════
// SLIDE N2: Repository Structure
// ══════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  addContentChrome(slide, "Repository Structure");

  // File tree on left as a styled code block
  const treeLines = [
    { text: "UCSF-PCNSL/\n", options: { bold: true, color: C.white, fontSize: 12.5, fontFace: "Consolas", breakLine: true } },
    { text: "  pcnsl_data_loader.py", options: { color: C.codeKw, fontSize: 11, fontFace: "Consolas", breakLine: true } },
    { text: "      Core module \u2014 1,500 lines\n", options: { color: C.codeComment, fontSize: 9.5, fontFace: "Calibri", italic: true, breakLine: true } },
    { text: "  __init__.py\n", options: { color: C.codeKw, fontSize: 11, fontFace: "Consolas", breakLine: true } },
    { text: "  combined_lesion_data.csv\n", options: { color: C.orange, fontSize: 11, fontFace: "Consolas", breakLine: true } },
    { text: "  data_dictionary_*.csv\n", options: { color: C.orange, fontSize: 11, fontFace: "Consolas", breakLine: true } },
    { text: "  figures_for_manuscript.ipynb\n", options: { color: C.codeStr, fontSize: 11, fontFace: "Consolas", breakLine: true } },
    { text: "  get-to-know-a-dataset-pcnsl.ipynb\n", options: { color: C.codeStr, fontSize: 11, fontFace: "Consolas", breakLine: true } },
    { text: "  .claude/skills/pcnsl-data-analysis/", options: { color: C.codeFn, fontSize: 11, fontFace: "Consolas" } },
  ];

  // Dark code panel (shortened to avoid footer collision)
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 0.95, w: 4.8, h: 3.4,
    fill: { color: C.codeBg },
    shadow: mkShadow(),
    rectRadius: 0.08,
  });
  slide.addText(treeLines, {
    x: 0.6, y: 1.1, w: 4.4, h: 3.1,
    valign: "top", margin: 0,
    paraSpaceAfter: 2,
  });

  // Legend on right
  const legendItems = [
    { color: C.codeKw, label: "Python Code" },
    { color: C.orange, label: "Data Files" },
    { color: C.codeStr, label: "Notebooks" },
    { color: C.codeFn, label: "AI Skill" },
  ];
  let ly = 1.1;
  for (const item of legendItems) {
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 5.6, y: ly, w: 0.25, h: 0.25, fill: { color: item.color },
      rectRadius: 0.04,
    });
    slide.addText(item.label, {
      x: 6.0, y: ly, w: 3.0, h: 0.25,
      fontSize: 13, fontFace: "Calibri", bold: true, color: C.navy,
      margin: 0, valign: "middle",
    });
    ly += 0.4;
  }

  // Summary stats on right (positioned to fit above footer)
  ly += 0.15;
  const stats = [
    { num: "2", label: "Loader Classes" },
    { num: "9", label: "Convenience Functions" },
    { num: "5", label: "Type Aliases" },
  ];
  for (const s of stats) {
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 5.6, y: ly, w: 3.6, h: 0.58,
      fill: { color: C.white }, shadow: mkShadow(),
    });
    slide.addText(s.num, {
      x: 5.7, y: ly, w: 0.7, h: 0.58,
      fontSize: 26, fontFace: "Georgia", bold: true, color: C.teal,
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText(s.label, {
      x: 6.4, y: ly, w: 2.7, h: 0.58,
      fontSize: 13, fontFace: "Calibri", color: C.body,
      valign: "middle", margin: 0,
    });
    ly += 0.7;
  }
}

// ══════════════════════════════════════════════════════════════
// SLIDE N3: Two Loaders, One Backend
// ══════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  addContentChrome(slide, "Two Loaders, One Backend");

  // Left card: PCNSLDataLoader (narrower to leave room for composition label)
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 1.05, w: 3.7, h: 3.55, fill: { color: C.white }, shadow: mkShadow(),
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 1.05, w: 3.7, h: 0.08, fill: { color: C.teal },
  });
  // Number circle
  slide.addShape(pres.shapes.OVAL, {
    x: 1.8, y: 1.25, w: 0.52, h: 0.52, fill: { color: C.teal },
  });
  slide.addText("1", {
    x: 1.8, y: 1.25, w: 0.52, h: 0.52,
    fontSize: 20, fontFace: "Georgia", bold: true, color: C.white,
    align: "center", valign: "middle", margin: 0,
  });
  slide.addText("PCNSLDataLoader", {
    x: 0.55, y: 1.9, w: 3.4, h: 0.35,
    fontSize: 16, fontFace: "Calibri", bold: true, color: C.navy,
    align: "center", margin: 0,
  });
  slide.addText("Core imaging I/O", {
    x: 0.55, y: 2.2, w: 3.4, h: 0.25,
    fontSize: 12, fontFace: "Calibri", italic: true, color: C.teal,
    align: "center", margin: 0,
  });
  const leftItems = [
    "Reads BIDS local filesystem",
    "Subject & session discovery",
    "Load anatomy, statistics, masks",
    "Returns NIfTI + DataFrames",
  ];
  slide.addText(
    leftItems.map((t, i) => ({
      text: t,
      options: { bullet: true, breakLine: i < leftItems.length - 1, color: C.body, fontSize: 11.5, fontFace: "Calibri" },
    })),
    { x: 0.7, y: 2.55, w: 3.2, h: 1.8, margin: 0, valign: "top",
      paraSpaceAfter: 4 }
  );

  // Right card: AWSDataLoader (narrower to leave room for composition label)
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.9, y: 1.05, w: 3.7, h: 3.55, fill: { color: C.white }, shadow: mkShadow(),
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.9, y: 1.05, w: 3.7, h: 0.08, fill: { color: C.navy },
  });
  slide.addShape(pres.shapes.OVAL, {
    x: 7.4, y: 1.25, w: 0.52, h: 0.52, fill: { color: C.navy },
  });
  slide.addText("2", {
    x: 7.4, y: 1.25, w: 0.52, h: 0.52,
    fontSize: 20, fontFace: "Georgia", bold: true, color: C.white,
    align: "center", valign: "middle", margin: 0,
  });
  slide.addText("AWSDataLoader", {
    x: 6.05, y: 1.9, w: 3.4, h: 0.35,
    fontSize: 16, fontFace: "Calibri", bold: true, color: C.navy,
    align: "center", margin: 0,
  });
  slide.addText("Clinical + imaging wrapper", {
    x: 6.05, y: 2.2, w: 3.4, h: 0.25,
    fontSize: 12, fontFace: "Calibri", italic: true, color: C.navy,
    align: "center", margin: 0,
  });
  const rightItems = [
    "Wraps PCNSLDataLoader internally",
    "6 clinical CSV types",
    "DICOM metadata parsing",
    "Patient-to-subject mapping",
  ];
  slide.addText(
    rightItems.map((t, i) => ({
      text: t,
      options: { bullet: true, breakLine: i < rightItems.length - 1, color: C.body, fontSize: 11.5, fontFace: "Calibri" },
    })),
    { x: 6.1, y: 2.55, w: 3.3, h: 1.8, margin: 0, valign: "top",
      paraSpaceAfter: 4 }
  );

  // Composition arrow (gold line between cards — wider gap now)
  slide.addShape(pres.shapes.LINE, {
    x: 4.15, y: 2.82, w: 1.7, h: 0,
    line: { color: C.gold, width: 2.5 },
  });
  slide.addText("composition", {
    x: 4.1, y: 2.5, w: 1.8, h: 0.3,
    fontSize: 10, fontFace: "Calibri", italic: true, color: C.gold,
    align: "center", margin: 0,
  });
}

// ══════════════════════════════════════════════════════════════
// SLIDE N4: Convenience Functions at a Glance
// ══════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  addContentChrome(slide, "Convenience Functions at a Glance");

  const tableRows = [
    [
      { text: "Research Question", options: { bold: true, color: C.white, fill: { color: C.navy }, fontSize: 11 } },
      { text: "Function", options: { bold: true, color: C.white, fill: { color: C.navy }, fontSize: 11 } },
      { text: "Returns", options: { bold: true, color: C.white, fill: { color: C.navy }, fontSize: 11 } },
    ],
    [
      { text: "Demographics + clinical", options: { fontSize: 10.5 } },
      { text: "load_aws_clinical_imaging_merged()", options: { fontSize: 9.5, fontFace: "Consolas", color: C.teal } },
      { text: "150 rows", options: { fontSize: 10.5 } },
    ],
    [
      { text: "Genomic mutations", options: { fontSize: 10.5 } },
      { text: "load_aws_mutations()", options: { fontSize: 9.5, fontFace: "Consolas", color: C.teal } },
      { text: "64-patient subset", options: { fontSize: 10.5 } },
    ],
    [
      { text: "Scanner parameters", options: { fontSize: 10.5 } },
      { text: "load_aws_dicom_headers()", options: { fontSize: 9.5, fontFace: "Consolas", color: C.teal } },
      { text: "600 rows", options: { fontSize: 10.5 } },
    ],
    [
      { text: "Voxel geometry", options: { fontSize: 10.5 } },
      { text: "load_aws_dicom_geometry()", options: { fontSize: 9.5, fontFace: "Consolas", color: C.teal } },
      { text: "600 rows", options: { fontSize: 10.5 } },
    ],
    [
      { text: "Lesion summary stats", options: { fontSize: 10.5 } },
      { text: "load_all_summary_statistics()", options: { fontSize: 9.5, fontFace: "Consolas", color: C.teal } },
      { text: "150 rows/modality", options: { fontSize: 10.5 } },
    ],
    [
      { text: "Per-lesion measurements", options: { fontSize: 10.5 } },
      { text: "load_all_individual_lesions()", options: { fontSize: 9.5, fontFace: "Consolas", color: C.teal } },
      { text: "Variable rows", options: { fontSize: 10.5 } },
    ],
    [
      { text: "Radiomics features", options: { fontSize: 10.5 } },
      { text: "load_all_radiomics()", options: { fontSize: 9.5, fontFace: "Consolas", color: C.teal } },
      { text: "150 rows/modality", options: { fontSize: 10.5 } },
    ],
  ];

  slide.addTable(tableRows, {
    x: 0.4, y: 1.0, w: 9.2,
    colW: [2.6, 4.2, 2.4],
    border: { type: "solid", pt: 0.5, color: "D0D0D0" },
    fontFace: "Calibri",
    color: C.body,
    rowH: [0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38],
    autoPage: false,
  });

  // Callout note at bottom
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 4.15, w: 9.2, h: 0.55,
    fill: { color: C.white }, shadow: mkShadow(),
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 4.15, w: 0.06, h: 0.55, fill: { color: C.gold },
  });
  slide.addText("All functions return pandas DataFrames.  Modality options: FLAIR or T1Post.", {
    x: 0.65, y: 4.15, w: 8.7, h: 0.55,
    fontSize: 11, fontFace: "Calibri", color: C.body,
    valign: "middle", margin: 0,
  });
}

// ══════════════════════════════════════════════════════════════
// SLIDE N5: Loading Data in 3 Lines
// ══════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  addContentChrome(slide, "Loading Data in 3 Lines");

  // Code block 1
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 0.95, w: 5.8, h: 1.65,
    fill: { color: C.codeBg }, shadow: mkShadow(), rectRadius: 0.06,
  });
  const code1 = [
    { text: "from ", options: { color: C.codeKw, fontSize: 11, fontFace: "Consolas", breakLine: false } },
    { text: "pcnsl_data_loader ", options: { color: C.codeText, fontSize: 11, fontFace: "Consolas", breakLine: false } },
    { text: "import ", options: { color: C.codeKw, fontSize: 11, fontFace: "Consolas", breakLine: true } },
    { text: "    load_aws_clinical_imaging_merged", options: { color: C.codeFn, fontSize: 11, fontFace: "Consolas", breakLine: true } },
    { text: "", options: { fontSize: 6, breakLine: true } },
    { text: "df = load_aws_clinical_imaging_merged()", options: { color: C.codeText, fontSize: 11, fontFace: "Consolas", breakLine: true } },
    { text: "print(f\"{len(df)} patients\")  ", options: { color: C.codeText, fontSize: 11, fontFace: "Consolas", breakLine: false } },
    { text: "# 150", options: { color: C.codeComment, fontSize: 11, fontFace: "Consolas" } },
  ];
  slide.addText(code1, { x: 0.65, y: 1.05, w: 5.3, h: 1.45, margin: 0, valign: "top" });

  // Annotation 1
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 6.45, y: 0.95, w: 3.2, h: 1.65,
    fill: { color: C.white }, shadow: mkShadow(),
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 6.45, y: 0.95, w: 0.06, h: 1.65, fill: { color: C.teal },
  });
  slide.addText("One function call returns merged clinical + imaging data for all 150 patients", {
    x: 6.7, y: 0.95, w: 2.8, h: 1.65,
    fontSize: 12, fontFace: "Calibri", color: C.body, valign: "middle", margin: 0,
  });

  // Code block 2
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 2.85, w: 5.8, h: 1.65,
    fill: { color: C.codeBg }, shadow: mkShadow(), rectRadius: 0.06,
  });
  const code2 = [
    { text: "from ", options: { color: C.codeKw, fontSize: 11, fontFace: "Consolas", breakLine: false } },
    { text: "pcnsl_data_loader ", options: { color: C.codeText, fontSize: 11, fontFace: "Consolas", breakLine: false } },
    { text: "import ", options: { color: C.codeKw, fontSize: 11, fontFace: "Consolas", breakLine: false } },
    { text: "AWSDataLoader", options: { color: C.codeFn, fontSize: 11, fontFace: "Consolas", breakLine: true } },
    { text: "", options: { fontSize: 6, breakLine: true } },
    { text: "loader = AWSDataLoader()", options: { color: C.codeText, fontSize: 11, fontFace: "Consolas", breakLine: true } },
    { text: "img, mask = loader.load_image_with_mask(", options: { color: C.codeText, fontSize: 11, fontFace: "Consolas", breakLine: false } },
    { text: "\"sub-0001\"", options: { color: C.codeStr, fontSize: 11, fontFace: "Consolas", breakLine: false } },
    { text: ")", options: { color: C.codeText, fontSize: 11, fontFace: "Consolas" } },
  ];
  slide.addText(code2, { x: 0.65, y: 2.95, w: 5.3, h: 1.45, margin: 0, valign: "top" });

  // Annotation 2
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 6.45, y: 2.85, w: 3.2, h: 1.65,
    fill: { color: C.white }, shadow: mkShadow(),
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 6.45, y: 2.85, w: 0.06, h: 1.65, fill: { color: C.navy },
  });
  slide.addText("Load NIfTI images with lesion overlays for any subject", {
    x: 6.7, y: 2.85, w: 2.8, h: 1.65,
    fontSize: 12, fontFace: "Calibri", color: C.body, valign: "middle", margin: 0,
  });
}

// ══════════════════════════════════════════════════════════════
// SLIDE N6: AI-Assisted Analysis with Claude Code
// ══════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  addContentChrome(slide, "AI-Assisted Analysis with Claude Code");

  const cards = [
    {
      num: "1", accent: C.teal,
      title: "Smart Data Routing",
      body: "Describe what you want to analyze. The skill maps your question to the right convenience function via a built-in function router.",
    },
    {
      num: "2", accent: C.navy,
      title: "9 Ready-Made Analysis Patterns",
      body: "Demographics, mutation frequency, lesion comparison, anatomical distribution, radiomics, scanner variability, acquisition parameters, NIfTI visualization, S3 access.",
    },
    {
      num: "3", accent: C.teal,
      title: "Built-In Gotcha Handling",
      body: "Automatic date century correction, GE field-strength conversion (Gauss to Tesla), SummaryLesions CSV transposition — common pitfalls handled for you.",
    },
  ];

  cards.forEach((c, i) => {
    const cx = 0.4 + i * 3.07;
    // Card bg
    slide.addShape(pres.shapes.RECTANGLE, {
      x: cx, y: 1.05, w: 2.85, h: 3.3, fill: { color: C.white }, shadow: mkShadow(),
    });
    // Accent top bar
    slide.addShape(pres.shapes.RECTANGLE, {
      x: cx, y: 1.05, w: 2.85, h: 0.08, fill: { color: c.accent },
    });
    // Number circle
    slide.addShape(pres.shapes.OVAL, {
      x: cx + 1.17, y: 1.3, w: 0.52, h: 0.52, fill: { color: c.accent },
    });
    slide.addText(c.num, {
      x: cx + 1.17, y: 1.3, w: 0.52, h: 0.52,
      fontSize: 20, fontFace: "Georgia", bold: true, color: C.white,
      align: "center", valign: "middle", margin: 0,
    });
    // Title
    slide.addText(c.title, {
      x: cx + 0.15, y: 1.95, w: 2.55, h: 0.45,
      fontSize: 14, fontFace: "Calibri", bold: true, color: C.navy,
      align: "center", valign: "middle", margin: 0,
    });
    // Body
    slide.addText(c.body, {
      x: cx + 0.2, y: 2.5, w: 2.45, h: 1.7,
      fontSize: 11, fontFace: "Calibri", color: C.body,
      valign: "top", margin: 0,
      lineSpacingMultiple: 1.35,
    });
  });
}

// ══════════════════════════════════════════════════════════════
// SLIDE N7: Analysis Patterns (3x3 grid)
// ══════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  addContentChrome(slide, "Skill Analysis Patterns");

  const patterns = [
    { title: "Demographics", desc: "Age, sex, race distributions" },
    { title: "Mutation Frequency", desc: "Top-N genes, variant allele freq" },
    { title: "Lesion Comparison", desc: "FLAIR vs T1Post volumes" },
    { title: "Anatomical Distribution", desc: "Regional % with bootstrapped CI" },
    { title: "Radiomics Features", desc: "Entropy, kurtosis, energy" },
    { title: "Scanner Variability", desc: "Manufacturer, field strength" },
    { title: "Acquisition Params", desc: "TR, TE, TI summary tables" },
    { title: "NIfTI Visualization", desc: "Orthographic + mosaic views" },
    { title: "S3 Direct Access", desc: "No-auth public bucket" },
  ];

  patterns.forEach((p, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const px = 0.4 + col * 3.07;
    const py = 0.95 + row * 1.2;
    const accent = i % 2 === 0 ? C.teal : C.navy;

    slide.addShape(pres.shapes.RECTANGLE, {
      x: px, y: py, w: 2.85, h: 1.0, fill: { color: C.white }, shadow: mkShadow(),
    });
    slide.addShape(pres.shapes.RECTANGLE, {
      x: px, y: py, w: 2.85, h: 0.36, fill: { color: accent },
    });
    slide.addText(p.title, {
      x: px + 0.12, y: py + 0.02, w: 2.61, h: 0.32,
      fontSize: 12, fontFace: "Calibri", bold: true, color: C.white,
      margin: 0, valign: "middle",
    });
    slide.addText(p.desc, {
      x: px + 0.12, y: py + 0.42, w: 2.61, h: 0.5,
      fontSize: 10.5, fontFace: "Calibri", color: C.body,
      margin: 0, valign: "middle",
    });
  });
}

// ══════════════════════════════════════════════════════════════
// SLIDE N8: Typical Analysis Workflow (5-step flow)
// ══════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  addContentChrome(slide, "Typical Analysis Workflow");

  const steps = [
    { num: "1", title: "Ask a Question", desc: "\"What is the mutation frequency of MYD88?\"" },
    { num: "2", title: "Skill Routes", desc: "Maps to load_aws_mutations()" },
    { num: "3", title: "Load & Fix", desc: "Century-correct dates, convert GE field strength" },
    { num: "4", title: "Analyze", desc: "Group by gene, compute frequencies, generate plots" },
    { num: "5", title: "Export", desc: "Publication-ready tables and figures" },
  ];

  steps.forEach((s, i) => {
    const sx = 0.2 + i * 1.82;
    // Card
    slide.addShape(pres.shapes.RECTANGLE, {
      x: sx, y: 1.15, w: 1.65, h: 2.5, fill: { color: C.white }, shadow: mkShadow(),
    });
    // Number circle
    slide.addShape(pres.shapes.OVAL, {
      x: sx + 0.56, y: 1.3, w: 0.52, h: 0.52, fill: { color: C.teal },
    });
    slide.addText(s.num, {
      x: sx + 0.56, y: 1.3, w: 0.52, h: 0.52,
      fontSize: 20, fontFace: "Georgia", bold: true, color: C.white,
      align: "center", valign: "middle", margin: 0,
    });
    // Title
    slide.addText(s.title, {
      x: sx + 0.08, y: 1.95, w: 1.49, h: 0.35,
      fontSize: 13, fontFace: "Calibri", bold: true, color: C.navy,
      align: "center", valign: "middle", margin: 0,
    });
    // Description
    slide.addText(s.desc, {
      x: sx + 0.08, y: 2.35, w: 1.49, h: 1.15,
      fontSize: 10, fontFace: "Calibri", color: C.body,
      align: "center", valign: "top", margin: 0,
      lineSpacingMultiple: 1.3,
    });
    // Connector arrow (gold line) — skip after last
    if (i < steps.length - 1) {
      slide.addShape(pres.shapes.LINE, {
        x: sx + 1.65, y: 2.4, w: 0.17, h: 0,
        line: { color: C.gold, width: 2.5 },
      });
    }
  });

  // Bottom callout
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 3.9, w: 9.2, h: 0.7,
    fill: { color: C.white }, shadow: mkShadow(),
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 3.9, w: 0.06, h: 0.7, fill: { color: C.gold },
  });
  slide.addText("The Claude skill handles steps 2\u20134 automatically based on your natural-language question.", {
    x: 0.7, y: 3.9, w: 8.7, h: 0.7,
    fontSize: 13, fontFace: "Calibri", bold: true, color: C.navy,
    valign: "middle", margin: 0,
  });
}

// ── Write output ──
pres.writeFile({ fileName: "/Users/mromano/research/UCSF-PCNSL/new_code_slides.pptx" })
  .then(() => console.log("Created new_code_slides.pptx"))
  .catch(err => console.error(err));
