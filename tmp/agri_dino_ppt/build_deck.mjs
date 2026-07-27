import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT_DIR = "/Users/abedin/Desktop/AG_Project/AG-Foundational-Model/tmp/agri_dino_ppt";
const FINAL_PPTX = "/Users/abedin/Desktop/AG_Project/AG-Foundational-Model/reports/Agri_DINO_Eighth_Grade_Presentation.pptx";

const W = 1280;
const H = 720;
const C = {
  navy: "#102A43",
  ink: "#172B4D",
  muted: "#52606D",
  light: "#F4F7F5",
  gray: "#D9E2EC",
  green: "#2E7D5B",
  green2: "#72B79A",
  paleGreen: "#E9F4EE",
  orange: "#D97732",
  paleOrange: "#FFF1E8",
  red: "#C94C4C",
  paleRed: "#FBEAEA",
  white: "#FFFFFF",
};

function addShape(slide, geometry, left, top, width, height, fill = "none", lineFill = "none", lineWidth = 0, name) {
  return slide.shapes.add({
    geometry,
    name,
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: lineFill, width: lineWidth },
  });
}

function addText(slide, text, left, top, width, height, style = {}, name) {
  const sh = addShape(slide, "textbox", left, top, width, height, "none", "none", 0, name);
  sh.text = text;
  sh.text.style = {
    typeface: "Arial",
    fontSize: 24,
    color: C.ink,
    alignment: "left",
    verticalAlignment: "top",
    autoFit: "shrinkText",
    wrap: "square",
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
    ...style,
  };
  return sh;
}

function addRule(slide, left, top, width, color = C.gray, thickness = 2) {
  return addShape(slide, "line", left, top, width, 0, "none", color, thickness);
}

function addFooter(slide, number) {
  addRule(slide, 72, 681, 1136, C.gray, 1);
  addText(slide, "AGRI-DINO  /  PROJECT PRESENTATION", 72, 691, 430, 16, {
    fontSize: 12,
    color: C.muted,
    bold: true,
  });
  addText(slide, String(number).padStart(2, "0"), 1160, 689, 48, 18, {
    fontSize: 12,
    color: C.muted,
    bold: true,
    alignment: "right",
  });
}

function addTitle(slide, kicker, title, subtitle = "") {
  addText(slide, kicker.toUpperCase(), 72, 44, 400, 22, {
    fontSize: 14,
    color: C.green,
    bold: true,
    letterSpacing: 1,
  });
  addText(slide, title, 72, 78, 1136, 64, {
    fontSize: 38,
    color: C.navy,
    bold: true,
  });
  if (subtitle) {
    addText(slide, subtitle, 72, 148, 1000, 34, {
      fontSize: 19,
      color: C.muted,
    });
  }
}

function setNotes(slide, body) {
  slide.speakerNotes.textFrame.setText(`[Sources]\n- Agri_DINO_CVPR_Paper.tex and Agri_DINO_CVPR_Paper.pdf (project method, results, and figures).\n\n${body}`);
  slide.speakerNotes.setVisible(true);
}

function addDot(slide, x, y, r, fill) {
  return addShape(slide, "ellipse", x, y, r, r, fill, "none", 0);
}

function addArrow(slide, left, top, width, height, fill = C.green) {
  return addShape(slide, "rightArrow", left, top, width, height, fill, "none", 0);
}

function addPill(slide, text, left, top, width, fill, color = C.ink) {
  const p = addShape(slide, "roundRect", left, top, width, 36, fill, "none", 0);
  p.text = text;
  p.text.style = {
    typeface: "Arial",
    fontSize: 16,
    bold: true,
    color,
    alignment: "center",
    verticalAlignment: "middle",
    autoFit: "shrinkText",
    insets: { left: 8, right: 8, top: 0, bottom: 0 },
  };
  return p;
}

function addLeafMark(slide, left, top, scale = 1) {
  // A simple editable mark used only as a small visual cue, not as data.
  addShape(slide, "ellipse", left, top, 82 * scale, 44 * scale, C.green, "none", 0);
  addShape(slide, "ellipse", left + 41 * scale, top + 17 * scale, 62 * scale, 34 * scale, C.green2, "none", 0);
  addRule(slide, left + 22 * scale, top + 37 * scale, 70 * scale, C.white, 2);
}

function addSimpleBarChart(slide, data, left, top, width, height, maxValue, unit = "%") {
  const labelW = 220;
  const barLeft = left + labelW;
  const barW = width - labelW - 70;
  const rowH = height / data.length;
  data.forEach((d, i) => {
    const y = top + i * rowH + 8;
    addText(slide, d.label, left, y + 3, labelW - 14, 28, {
      fontSize: 18,
      color: C.ink,
      bold: d.highlight,
      verticalAlignment: "middle",
    });
    addShape(slide, "roundRect", barLeft, y, barW, 28, C.light, "none", 0);
    const fill = d.highlight ? C.green : d.color || C.navy;
    addShape(slide, "roundRect", barLeft, y, Math.max(12, (d.value / maxValue) * barW), 28, fill, "none", 0);
    addText(slide, `${d.value.toFixed(2)}${unit}`, barLeft + barW + 12, y + 2, 70, 28, {
      fontSize: 17,
      color: fill,
      bold: true,
      alignment: "right",
      verticalAlignment: "middle",
    });
  });
}

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  const p = Presentation.create({ slideSize: { width: W, height: H } });

  // 1. Title
  {
    const s = p.slides.add();
    s.background.fill = C.white;
    addShape(s, "rect", 0, 0, 20, H, C.green, "none", 0);
    addText(s, "A SIMPLE IDEA FOR\nBETTER PLANT RECOGNITION", 72, 92, 720, 150, {
      fontSize: 48,
      bold: true,
      color: C.navy,
      lineSpacing: 0.92,
    }, "title");
    addText(s, "Agri-DINO", 72, 272, 720, 68, {
      fontSize: 42,
      bold: true,
      color: C.green,
    });
    addText(s, "Teaching a computer to notice the small details in plant images", 72, 350, 620, 64, {
      fontSize: 24,
      color: C.muted,
    });
    addText(s, "Md Min-Ha-Zul Abedin  ·  Md Mehedi Hasan\nAuburn University", 72, 570, 520, 46, {
      fontSize: 18,
      color: C.ink,
      lineSpacing: 1.12,
    });
    addShape(s, "roundRect", 860, 120, 300, 430, C.paleGreen, "none", 0);
    addLeafMark(s, 916, 208, 1.65);
    addText(s, "1.52M", 902, 322, 220, 64, {
      fontSize: 54,
      bold: true,
      color: C.green,
      alignment: "center",
    });
    addText(s, "unlabeled plant\nphotos used to learn", 898, 398, 224, 66, {
      fontSize: 22,
      color: C.ink,
      alignment: "center",
      lineSpacing: 1.05,
    });
    setNotes(s, "Open with the everyday problem: plants can be hard to identify when the view, light, or background changes.");
  }

  // 2. Problem
  {
    const s = p.slides.add();
    s.background.fill = C.white;
    addTitle(s, "01  /  THE PROBLEM", "A plant can look different without changing its name", "A computer must learn the important clues, not the background noise.");
    const y = 250;
    const boxW = 292;
    const xs = [72, 416, 760];
    const labels = ["Different light", "Different angle", "Busy background"];
    const subs = ["Shadows can hide spots.", "Leaves can look smaller.", "Soil and tools can distract."];
    const fills = [C.paleOrange, C.paleGreen, C.paleRed];
    const colors = [C.orange, C.green, C.red];
    xs.forEach((x, i) => {
      addShape(s, "roundRect", x, y, boxW, 240, fills[i], "none", 0);
      addShape(s, "ellipse", x + 104, y + 36, 84, 84, colors[i], "none", 0);
      addShape(s, "ellipse", x + 126, y + 55, 58, 26, C.white, "none", 0);
      addRule(s, x + 114, y + 78, 70, C.white, 2);
      addText(s, labels[i], x + 24, y + 146, boxW - 48, 28, {
        fontSize: 22,
        bold: true,
        color: C.navy,
        alignment: "center",
      });
      addText(s, subs[i], x + 24, y + 181, boxW - 48, 34, {
        fontSize: 17,
        color: C.muted,
        alignment: "center",
      });
    });
    addText(s, "The hard part is not only naming a plant. It is recognizing the same plant in many real-world situations.", 160, 565, 960, 46, {
      fontSize: 23,
      color: C.ink,
      alignment: "center",
      bold: true,
    });
    addFooter(s, 2);
    setNotes(s, "Use the three examples to make domain shift concrete before introducing the model.");
  }

  // 3. Big idea
  {
    const s = p.slides.add();
    s.background.fill = C.white;
    addTitle(s, "02  /  THE BIG IDEA", "Let the model learn from many plant photos before the test", "It first learns what plant images usually look like, then learns a smaller job.");
    addShape(s, "roundRect", 72, 250, 430, 250, C.paleGreen, "none", 0);
    addText(s, "STEP 1", 104, 278, 130, 24, { fontSize: 16, bold: true, color: C.green });
    addText(s, "Learn from\nunlabeled photos", 104, 316, 310, 72, { fontSize: 33, bold: true, color: C.navy });
    addText(s, "No answer key is needed.\nThe model compares views\nof the same picture.", 104, 410, 300, 64, { fontSize: 20, color: C.muted, lineSpacing: 1.08 });
    addArrow(s, 548, 340, 148, 70, C.green);
    addShape(s, "roundRect", 748, 250, 430, 250, C.paleOrange, "none", 0);
    addText(s, "STEP 2", 780, 278, 130, 24, { fontSize: 16, bold: true, color: C.orange });
    addText(s, "Learn a\nsmall task", 780, 316, 310, 72, { fontSize: 33, bold: true, color: C.navy });
    addText(s, "Now use a small labeled\ndataset to answer a question,\nsuch as: “Which plant is this?”", 780, 410, 340, 64, { fontSize: 20, color: C.muted, lineSpacing: 1.08 });
    addText(s, "This is like studying a big picture book before taking a short quiz.", 185, 568, 910, 44, { fontSize: 24, color: C.ink, alignment: "center", bold: true });
    addFooter(s, 3);
    setNotes(s, "Explain self-supervised learning with the picture-book analogy. Keep the focus on learning first, testing later.");
  }

  // 4. Method
  {
    const s = p.slides.add();
    s.background.fill = C.white;
    addTitle(s, "03  /  HOW AGRI-DINO LEARNS", "The model looks at the same image in different ways", "Different crops and views help it focus on the plant, not one exact picture.");
    addText(s, "One plant photo", 92, 246, 190, 28, { fontSize: 20, bold: true, color: C.navy, alignment: "center" });
    addShape(s, "roundRect", 90, 292, 190, 190, C.paleGreen, C.green2, 2);
    addShape(s, "ellipse", 132, 350, 100, 54, C.green, "none", 0);
    addShape(s, "ellipse", 174, 372, 82, 42, C.green2, "none", 0);
    addRule(s, 150, 386, 95, C.white, 2);
    addPill(s, "crop", 116, 514, 70, C.paleGreen, C.green);
    addPill(s, "zoom", 196, 514, 70, C.paleGreen, C.green);
    addArrow(s, 330, 356, 100, 52, C.green);
    addShape(s, "roundRect", 468, 248, 266, 280, C.light, C.gray, 1);
    addText(s, "Two learners compare\nthe views", 496, 270, 210, 56, { fontSize: 24, bold: true, color: C.navy, alignment: "center" });
    addShape(s, "ellipse", 522, 360, 76, 76, C.green, "none", 0);
    addText(s, "student", 502, 443, 116, 24, { fontSize: 18, color: C.green, bold: true, alignment: "center" });
    addShape(s, "ellipse", 612, 360, 76, 76, C.orange, "none", 0);
    addText(s, "teacher", 592, 443, 116, 24, { fontSize: 18, color: C.orange, bold: true, alignment: "center" });
    addRule(s, 594, 397, 24, C.navy, 3);
    addText(s, "Are the important clues\nstill the same?", 504, 475, 194, 38, { fontSize: 17, color: C.muted, alignment: "center" });
    addArrow(s, 790, 356, 100, 52, C.green);
    addShape(s, "roundRect", 948, 248, 250, 280, C.paleOrange, "none", 0);
    addText(s, "Better plant\nfeatures", 980, 310, 186, 72, { fontSize: 32, bold: true, color: C.navy, alignment: "center" });
    addText(s, "The model keeps\nuseful patterns and\nignores distractions.", 980, 408, 186, 66, { fontSize: 20, color: C.muted, alignment: "center", lineSpacing: 1.05 });
    addFooter(s, 4);
    setNotes(s, "The technical names are not needed here: student and teacher are simply two views that help the model check its own learning.");
  }

  // 5. Adaptation
  {
    const s = p.slides.add();
    s.background.fill = C.white;
    addTitle(s, "04  /  ADAPTING TO A NEW TASK", "Warm up first, then fine-tune carefully", "The two-step training recipe helps a small plant dataset work with a large model.");
    addText(s, "A large model already knows useful visual patterns.", 110, 220, 1060, 34, { fontSize: 23, color: C.ink, alignment: "center", bold: true });
    addShape(s, "roundRect", 92, 310, 300, 170, C.paleGreen, "none", 0);
    addText(s, "1", 120, 332, 52, 52, { fontSize: 42, bold: true, color: C.green, alignment: "center" });
    addText(s, "Warm up the\nnew answer head", 190, 332, 168, 58, { fontSize: 26, bold: true, color: C.navy });
    addText(s, "Keep the main model still\nfor a few epochs.", 120, 416, 230, 42, { fontSize: 18, color: C.muted, alignment: "center" });
    addArrow(s, 432, 360, 120, 50, C.green);
    addShape(s, "roundRect", 620, 310, 300, 170, C.paleOrange, "none", 0);
    addText(s, "2", 648, 332, 52, 52, { fontSize: 42, bold: true, color: C.orange, alignment: "center" });
    addText(s, "Fine-tune\nall layers gently", 718, 332, 168, 58, { fontSize: 26, bold: true, color: C.navy });
    addText(s, "Later layers change more;\nearly layers change less.", 648, 416, 230, 42, { fontSize: 18, color: C.muted, alignment: "center" });
    addArrow(s, 960, 360, 120, 50, C.orange);
    addShape(s, "roundRect", 1132, 310, 82, 170, C.paleRed, "none", 0);
    addText(s, "✓", 1144, 340, 60, 60, { fontSize: 46, bold: true, color: C.red, alignment: "center" });
    addText(s, "test", 1145, 420, 58, 28, { fontSize: 19, bold: true, color: C.red, alignment: "center" });
    addText(s, "We also test a flipped version of the image and average the answers.", 204, 568, 872, 44, { fontSize: 23, color: C.ink, alignment: "center", bold: true });
    addFooter(s, 5);
    setNotes(s, "This slide explains LP-FT and LLRD without relying on acronyms. The key idea is to protect useful old knowledge while learning the new task.");
  }

  // 6. Main result
  {
    const s = p.slides.add();
    s.background.fill = C.white;
    addTitle(s, "05  /  MAIN RESULT", "Pretraining gave the model a 10-point jump", "On the MedicinalPlant test, Agri-DINO was almost as accurate as supervised ImageNet pretraining.");
    addSimpleBarChart(s, [
      { label: "Random start", value: 49.35, color: C.red },
      { label: "Agri-DINO", value: 59.67, highlight: true },
      { label: "ImageNet (supervised)", value: 59.99, color: C.navy },
    ], 118, 244, 760, 210, 65);
    addShape(s, "roundRect", 928, 244, 250, 215, C.paleGreen, "none", 0);
    addText(s, "+10.32", 960, 278, 184, 60, { fontSize: 44, bold: true, color: C.green, alignment: "center" });
    addText(s, "percentage points\nabove a random start", 960, 350, 184, 48, { fontSize: 20, color: C.ink, alignment: "center", lineSpacing: 1.05 });
    addText(s, "Why this matters", 120, 520, 240, 28, { fontSize: 22, bold: true, color: C.navy });
    addText(s, "The model learned useful plant patterns before it ever saw the small test set's labels.", 120, 556, 950, 44, { fontSize: 23, color: C.ink });
    addFooter(s, 6);
    setNotes(s, "Point out that Agri-DINO is 0.32 points below the supervised ImageNet baseline, so the claim is competitive transfer, not a victory over every baseline.");
  }

  // 7. Ablation
  {
    const s = p.slides.add();
    s.background.fill = C.white;
    addTitle(s, "06  /  WHAT HELPED MOST", "Small training choices made a big difference", "The biggest jump came from the warm-up step before full fine-tuning.");
    const steps = [
      ["No warm-up", 35.19, C.red],
      ["Higher resolution", 39.82, C.orange],
      ["+ warm-up", 52.41, C.green],
      ["+ gentle updates", 57.88, C.green2],
      ["+ flip test", 59.67, C.navy],
    ];
    const x0 = 112;
    const chartTop = 258;
    const chartH = 260;
    const barW = 150;
    const gap = 40;
    const maxV = 65;
    addRule(s, x0, chartTop + chartH, 930, C.gray, 1);
    [0, 20, 40, 60].forEach((v) => {
      const yy = chartTop + chartH - (v / maxV) * chartH;
      addRule(s, x0, yy, 930, C.gray, 1);
      addText(s, String(v), 72, yy - 9, 30, 20, { fontSize: 14, color: C.muted, alignment: "right" });
    });
    steps.forEach((d, i) => {
      const x = x0 + i * (barW + gap);
      const bh = (d[1] / maxV) * chartH;
      addShape(s, "roundRect", x, chartTop + chartH - bh, barW, bh, d[2], "none", 0);
      addText(s, d[1].toFixed(2), x, chartTop + chartH - bh - 34, barW, 28, { fontSize: 18, bold: true, color: d[2], alignment: "center" });
      addText(s, d[0], x - 8, chartTop + chartH + 18, barW + 16, 48, { fontSize: 16, color: C.ink, alignment: "center", lineSpacing: 1.0 });
    });
    addShape(s, "roundRect", 1010, 270, 210, 220, C.paleOrange, "none", 0);
    addText(s, "+12.59", 1032, 304, 166, 54, { fontSize: 40, bold: true, color: C.orange, alignment: "center" });
    addText(s, "points from\nthe warm-up", 1032, 370, 166, 52, { fontSize: 23, color: C.navy, alignment: "center", lineSpacing: 1.05 });
    addText(s, "The recipe is not magic. The order of the steps matters.", 170, 588, 930, 38, { fontSize: 23, color: C.ink, bold: true, alignment: "center" });
    addFooter(s, 7);
    setNotes(s, "Stress that this is a cumulative experiment: each bar adds one ingredient to the previous setting.");
  }

  // 8. Conclusion
  {
    const s = p.slides.add();
    s.background.fill = C.white;
    addTitle(s, "07  /  TAKEAWAYS", "A simple recipe can help computers read plant images", "The project gives us a useful starting point—and a reminder to check the data carefully.");
    const xs = [90, 430, 770];
    const heads = ["Learn first", "Adapt gently", "Check labels"];
    const bodies = [
      "Use many unlabeled plant photos to learn visual patterns.",
      "Warm up the new task, then update the model carefully.",
      "A confusing label system can make a good model look bad.",
    ];
    const fills = [C.paleGreen, C.paleOrange, C.paleRed];
    const colors = [C.green, C.orange, C.red];
    xs.forEach((x, i) => {
      addShape(s, "roundRect", x, 250, 270, 220, fills[i], "none", 0);
      addDot(s, x + 26, 280, 22, colors[i]);
      addText(s, heads[i], x + 62, 274, 180, 30, { fontSize: 23, bold: true, color: C.navy });
      addText(s, bodies[i], x + 26, 338, 218, 86, { fontSize: 19, color: C.ink, lineSpacing: 1.08 });
    });
    addShape(s, "roundRect", 160, 532, 960, 72, C.navy, "none", 0);
    addText(s, "Big idea: better examples + careful training = better plant recognition.", 190, 552, 900, 32, { fontSize: 24, bold: true, color: C.white, alignment: "center" });
    addText(s, "Questions?", 72, 640, 400, 30, { fontSize: 22, color: C.green, bold: true });
    addFooter(s, 8);
    setNotes(s, "Close by returning to the opening idea: the model should recognize the same plant even when the picture changes.");
  }

  for (const [i, slide] of p.slides.items.entries()) {
    const stem = `slide-${String(i + 1).padStart(2, "0")}`;
    await writeBlob(`${OUT_DIR}/${stem}.png`, await p.export({ slide, format: "png", scale: 1 }));
    await fs.writeFile(`${OUT_DIR}/${stem}.layout.json`, await (await slide.export({ format: "layout" })).text());
  }
  await writeBlob(`${OUT_DIR}/deck-montage.webp`, await p.export({ format: "webp", montage: true, scale: 1 }));
  const pptx = await PresentationFile.exportPptx(p);
  await pptx.save(FINAL_PPTX);
  await fs.writeFile(`${OUT_DIR}/source-notes.txt`, "Project source: reports/Agri_DINO_CVPR_Paper.tex and reports/Agri_DINO_CVPR_Paper.pdf. No external assets used.\n");
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});

