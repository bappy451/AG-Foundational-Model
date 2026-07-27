import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const WORKSPACE = "/Users/abedin/Desktop/AG_Project/AG-Foundational-Model/tmp/agri_dino_ppt_revision";
const STARTER = `${WORKSPACE}/template-starter.pptx`;
const FINAL = "/Users/abedin/Desktop/AG_Project/AG-Foundational-Model/reports/Agri_DINO_Eighth_Grade_Presentation.pptx";

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  const presentation = await PresentationFile.importPptx(await FileBlob.load(STARTER));

  const before = await presentation.inspect({
    kind: "slide,textbox,shape,notes,layout",
    target: { id: "sl/y90nupkv", beforeLines: 0, afterLines: 20 },
    maxChars: 10000,
  });
  await fs.writeFile(`${WORKSPACE}/before-edit-slide-1.ndjson`, before.ndjson);

  const title = presentation.resolve("sh/547294r6");
  title.text = "Agri-DINO:\nDomain-adaptive self-supervision";

  const accent = presentation.resolve("sh/k3yl0zql");
  accent.text = "Fine-grained plant recognition";

  const method = presentation.resolve("sh/7qp4be9c");
  method.text = "Method: self-supervised pretraining + two-stage fine-tuning";

  const metric = presentation.resolve("sh/wn6dc7eh");
  metric.text = "1.52M";

  const result = presentation.resolve("sh/hofulsf2");
  result.text = "unlabeled photos\n59.67% top-1 accuracy";

  const notes = presentation.resolve("nt/y90nupkv");
  notes.setText(
    "[Sources]\n- Agri_DINO_CVPR_Paper.tex and Agri_DINO_CVPR_Paper.pdf (project method, results, and figures).\n\nFormal opening: introduce Agri-DINO as a self-supervised plant-recognition project. The cover summarizes the method and headline result; the following slides explain the problem, training recipe, and evidence."
  );
  notes.setVisible(true);

  const slide = presentation.resolve("sl/y90nupkv");
  await writeBlob(`${WORKSPACE}/after-edit-slide-1.png`, await slide.export({ format: "png", scale: 2 }));
  await fs.writeFile(`${WORKSPACE}/after-edit-slide-1.layout.json`, await (await slide.export({ format: "layout" })).text());

  const after = await presentation.inspect({
    kind: "slide,textbox,shape,notes,layout",
    target: { id: "sl/y90nupkv", beforeLines: 0, afterLines: 20 },
    maxChars: 10000,
  });
  await fs.writeFile(`${WORKSPACE}/after-edit-inspect.ndjson`, after.ndjson);

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
