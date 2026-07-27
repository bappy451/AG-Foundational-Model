import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const WORKSPACE = "/Users/abedin/Desktop/AG_Project/AG-Foundational-Model/tmp/agri_dino_ppt_technical";
const STARTER = `${WORKSPACE}/template-starter.pptx`;
const FINAL = "/Users/abedin/Desktop/AG_Project/AG-Foundational-Model/reports/Agri_DINO_Eighth_Grade_Presentation.pptx";

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

function setText(presentation, id, value) {
  presentation.resolve(id).text = value;
}

async function main() {
  const presentation = await PresentationFile.importPptx(await FileBlob.load(STARTER));

  const before = await presentation.inspect({
    kind: "slide,textbox,shape,notes,layout",
    maxChars: 24000,
  });
  await fs.writeFile(`${WORKSPACE}/before-edit-inspect.ndjson`, before.ndjson);

  // Slide 1: technical method label on the formal cover.
  setText(presentation, "sh/7qp4be9c", "Method: DINO-style pretraining + LP-FT/LLRD");

  // Slide 3: make the two-stage story technically concrete.
  setText(presentation, "sh/d0jax03i", "DINO-style self-supervision: learn features first, labels later.");
  setText(presentation, "sh/obq90bml", "Pretrain on\n1.52M photos");
  setText(presentation, "sh/pcjqtg36", "No human labels. Use two global + four local crops of each image.");
  setText(presentation, "sh/m1c3mlsn", "Fine-tune for\na target task");
  setText(presentation, "sh/943mhgre", "MedicinalPlant: 1,301 images, 30 species. Labels are used here.");
  setText(presentation, "sh/83ulovat", "Transfer learning: general features first, task-specific features second.");

  // Slide 4: name the backbone and the DINO-style comparison.
  setText(presentation, "sh/1cj2d8b6", "DINO compares different views of the same image");
  setText(presentation, "sh/0ba143al", "Student–teacher matching with a ViT-B/16 backbone.");
  setText(presentation, "sh/rm1k7yt4", "Multi-crop views");
  setText(presentation, "sh/jadsz2xk", "Student + EMA\nteacher");
  setText(presentation, "sh/h4fa9wfy", "Cross-view loss encourages similar features.");
  setText(presentation, "sh/3ytsrmpw", "Agri-DINO\nfeatures");
  setText(presentation, "sh/2xkrih8b", "2 global (224²) + 4 local (96²) views help it focus on plant structure.");

  // Slide 5: expose the adaptation protocol without overwhelming the audience.
  setText(presentation, "sh/dgbulwnm", "LP-FT + LLRD: warm up, then fine-tune");
  setText(presentation, "sh/cf2tcr61", "A structured recipe protects pretrained features while adapting to MedicinalPlant.");
  setText(presentation, "sh/z2tcnm5s", "Start from a DINOv3-initialized ViT-B/16.");
  setText(presentation, "sh/032tgr6d", "Linear probe\n(LP)");
  setText(presentation, "sh/n6dcr65o", "5 epochs; backbone frozen; train only the classifier.");
  setText(presentation, "sh/ahkvi1cb", "Fine-tune with\nLLRD");
  setText(presentation, "sh/v2twb6dw", "All layers update; γ = 0.75 gives smaller changes to early blocks.");
  setText(presentation, "sh/j6dwfqds", "TTA");
  setText(presentation, "sh/sb6xsvu9", "TTA averages the original and horizontally flipped views.");

  // Slide 6: identify the evaluation protocol behind the main chart.
  setText(presentation, "sh/zi98nu94", "MedicinalPlant: 1,301 images, 30 species, fixed stratified 80/20 split.");
  setText(presentation, "sh/cbe5g3ih", "No human labels were used during pretraining; labels were used only for downstream tuning.");

  // Slide 7: make the ablation labels explicit.
  setText(presentation, "sh/cb2tkvap", "Cumulative ablation on MedicinalPlant (256×256 input).");
  setText(presentation, "sh/dcbud0ra", "Each bar adds one component to the previous setting.");
  setText(presentation, "sh/q143ylov", "+ LLRD (γ = 0.75)");

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(`${WORKSPACE}/after-${stem}.png`, await presentation.export({ slide, format: "png", scale: 2 }));
    await fs.writeFile(`${WORKSPACE}/after-${stem}.layout.json`, await (await slide.export({ format: "layout" })).text());
  }

  const notes = presentation.resolve("nt/y90nupkv");
  notes.setText(
    "[Sources]\n- Agri_DINO_CVPR_Paper.tex and Agri_DINO_CVPR_Paper.pdf (project method, results, and figures).\n\nFormal opening: introduce Agri-DINO as a DINO-style self-supervised project. The cover now previews the backbone adaptation recipe and headline result; the following slides explain the details."
  );
  notes.setVisible(true);

  const after = await presentation.inspect({ kind: "slide,textbox,shape,notes,layout", maxChars: 24000 });
  await fs.writeFile(`${WORKSPACE}/after-edit-inspect.ndjson`, after.ndjson);

  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await writeBlob(`${WORKSPACE}/after-montage.webp`, montage);
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

