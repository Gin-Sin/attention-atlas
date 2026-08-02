#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const EXPECTED_IDS = [
  "mha",
  "mqa",
  "gqa",
  "mla",
  "dsa",
  "csa",
  "hca",
  "linear",
  "gated-delta",
  "kda"
];
const EXPECTED_SOURCE_PATHS = new Map([
  ["mha", "pytorch/mha.py"],
  ["mqa", "pytorch/mqa.py"],
  ["gqa", "pytorch/gqa.py"],
  ["mla", "pytorch/mla.py"],
  ["dsa", "pytorch/dsa.py"],
  ["csa", "pytorch/csa.py"],
  ["hca", "pytorch/hca.py"],
  ["linear", "pytorch/linear_attention.py"],
  ["gated-delta", "pytorch/gated_delta.py"],
  ["kda", "pytorch/kda.py"]
]);
const VALID_CATEGORIES = new Set(["dense", "sparse", "linear", "hybrid"]);
const POSITION_FIELDS = ["title", "summary", "equation", "caveat"];
const POSITION_STEP_FIELDS = ["label", "title", "body"];
const DERIVATION_FIELDS = ["title", "body", "source"];
const EXERCISE_FIELDS = ["kind", "level", "q", "hint", "answer"];
const OPEN_MARKER = /^\s*# \[Block (\d{2})\] (.+?)\s*$/;
const CLOSE_MARKER = /^\s*# \[\/Block (\d{2})\]\s*$/;
const MARKER_FRAGMENT = /# \[\/?Block\b/;

const errors = [];
const counts = {
  chapters: 0,
  derivations: 0,
  exercises: 0,
  diagrams: 0,
  interactiveNodes: 0,
  implementations: 0,
  blocks: 0,
  indexLinks: 0
};

function addError(message) {
  errors.push(message);
}

function readText(relativePath) {
  return fs.readFileSync(path.join(ROOT, relativePath), "utf8");
}

function isNonEmptyString(value) {
  return typeof value === "string" && value.trim().length > 0;
}

function requireString(record, field, location) {
  if (!record || !isNonEmptyString(record[field])) {
    addError(`${location}.${field} must be a nonempty string`);
    return false;
  }
  return true;
}

function compareOrderedIds(actual, expected, location) {
  if (actual.length !== expected.length) {
    addError(
      `${location} must contain exactly ${expected.length} ids; found ${actual.length}`
    );
  }
  const longest = Math.max(actual.length, expected.length);
  for (let index = 0; index < longest; index += 1) {
    if (actual[index] !== expected[index]) {
      addError(
        `${location}[${index}] must be ${JSON.stringify(expected[index])}; ` +
          `found ${JSON.stringify(actual[index])}`
      );
    }
  }
}

function createBrowserContext() {
  const sandbox = Object.create(null);
  sandbox.window = Object.create(null);
  return vm.createContext(sandbox, {
    name: "attention-atlas-content-validator",
    codeGeneration: { strings: false, wasm: false }
  });
}

function executeBrowserAsset(context, relativePath) {
  const source = readText(relativePath);
  const script = new vm.Script(source, {
    filename: path.join(ROOT, relativePath),
    displayErrors: true
  });
  script.runInContext(context, { timeout: 2_000 });
  return source;
}

function validateChapters(chapters) {
  if (!Array.isArray(chapters)) {
    addError("assets/chapters.js must assign an array to window.ATTENTION_CHAPTERS");
    return;
  }

  counts.chapters = chapters.length;
  const ids = chapters.map((chapter) => chapter && chapter.id);
  compareOrderedIds(ids, EXPECTED_IDS, "chapter ids");

  const uniqueIds = new Set(ids);
  if (uniqueIds.size !== ids.length) {
    addError("chapter ids must be unique");
  }

  chapters.forEach((chapter, index) => {
    const location = `chapter[${index}]${chapter && chapter.id ? ` (${chapter.id})` : ""}`;
    if (!chapter || typeof chapter !== "object") {
      addError(`${location} must be an object`);
      return;
    }

    if (!Number.isInteger(chapter.order) || chapter.order !== index) {
      addError(`${location}.order must be the contiguous integer ${index}`);
    }
    if (!VALID_CATEGORIES.has(chapter.category)) {
      addError(
        `${location}.category must be one of ${Array.from(VALID_CATEGORIES).join(", ")}; ` +
          `found ${JSON.stringify(chapter.category)}`
      );
    }

    const position = chapter.positionEncoding;
    if (!position || typeof position !== "object") {
      addError(`${location}.positionEncoding must be an object`);
    } else {
      POSITION_FIELDS.forEach((field) => {
        requireString(position, field, `${location}.positionEncoding`);
      });
      if (!Array.isArray(position.steps) || position.steps.length !== 3) {
        addError(`${location}.positionEncoding.steps must contain exactly 3 steps`);
      } else {
        position.steps.forEach((step, stepIndex) => {
          POSITION_STEP_FIELDS.forEach((field) => {
            requireString(
              step,
              field,
              `${location}.positionEncoding.steps[${stepIndex}]`
            );
          });
        });
      }
    }

    if (
      !Array.isArray(chapter.derivations) ||
      chapter.derivations.length < 4 ||
      chapter.derivations.length > 6
    ) {
      addError(`${location}.derivations must contain 4–6 entries`);
    } else {
      counts.derivations += chapter.derivations.length;
      chapter.derivations.forEach((derivation, derivationIndex) => {
        DERIVATION_FIELDS.forEach((field) => {
          requireString(
            derivation,
            field,
            `${location}.derivations[${derivationIndex}]`
          );
        });
      });
    }

    if (!Array.isArray(chapter.exercises) || chapter.exercises.length !== 6) {
      addError(`${location}.exercises must contain exactly 6 entries`);
    } else {
      counts.exercises += chapter.exercises.length;
      chapter.exercises.forEach((exercise, exerciseIndex) => {
        EXERCISE_FIELDS.forEach((field) => {
          requireString(
            exercise,
            field,
            `${location}.exercises[${exerciseIndex}]`
          );
        });
      });
    }

    if (!Array.isArray(chapter.sources) || chapter.sources.length === 0) {
      addError(`${location}.sources must be a nonempty array`);
    } else {
      chapter.sources.forEach((source, sourceIndex) => {
        const sourceLocation = `${location}.sources[${sourceIndex}]`;
        requireString(source, "label", sourceLocation);
        if (!requireString(source, "url", sourceLocation)) return;
        try {
          const parsed = new URL(source.url);
          if (
            !["http:", "https:"].includes(parsed.protocol) ||
            !parsed.hostname ||
            parsed.username ||
            parsed.password
          ) {
            throw new Error("expected an absolute HTTP(S) URL without credentials");
          }
        } catch (error) {
          addError(`${sourceLocation}.url is invalid (${error.message}): ${source.url}`);
        }
      });
    }
  });
}

function validateDiagrams(chapters, diagramBuilder, implementations) {
  if (!diagramBuilder || typeof diagramBuilder.build !== "function") {
    addError("assets/diagrams.js must expose window.AttentionDiagrams.build");
    return;
  }
  if (!Array.isArray(chapters)) return;

  chapters.forEach((chapter) => {
    const location = `diagram for ${chapter.id}`;
    if (!chapter.diagram || typeof chapter.diagram !== "object") {
      addError(`${location} must have a configuration object`);
      return;
    }
    try {
      const report = diagramBuilder.build(chapter.diagram);
      if (!report || typeof report !== "object") {
        addError(`${location} build must return an object`);
        return;
      }
      if (!isNonEmptyString(report.svg) || !report.svg.includes("<svg")) {
        addError(`${location} build returned an empty or invalid svg`);
      }
      const validBlocks = new Set(
        (implementations?.[chapter.id]?.blocks || []).map((block) => String(block.id))
      );
      const interactiveBlocks = Array.from(
        report.svg.matchAll(/\bdata-code-block="([^"]+)"/g),
        (match) => match[1]
      );
      if (interactiveBlocks.length === 0) {
        addError(`${location} has no interactive architecture-to-code nodes`);
      }
      interactiveBlocks.forEach((blockId) => {
        if (!validBlocks.has(blockId)) {
          addError(`${location} links to unknown implementation block ${blockId}`);
        }
      });
      counts.interactiveNodes += interactiveBlocks.length;
      if (!Array.isArray(report.notes) || report.notes.length === 0) {
        addError(`${location} build returned no notes`);
      } else {
        report.notes.forEach((note, noteIndex) => {
          if (
            !Array.isArray(note) ||
            note.length < 2 ||
            !isNonEmptyString(note[0]) ||
            !isNonEmptyString(note[1])
          ) {
            addError(`${location} note[${noteIndex}] must contain a title and body`);
          }
        });
      }
      if (!isNonEmptyString(report.memory)) {
        addError(`${location} build returned an empty memory aid`);
      }
      counts.diagrams += 1;
    } catch (error) {
      addError(`${location} failed to build: ${error.message}`);
    }
  });
}

function validateRenderer(courseSource, chapterHtml) {
  if (!/richText\(d\.body\)/.test(courseSource)) {
    addError("assets/course.js must pass every derivation body through richText()");
  }
  if (!/data-architecture-ide/.test(courseSource) || !/data-workbench-editor/.test(courseSource)) {
    addError("assets/course.js must render the integrated diagram/code workbench");
  }
  if (/PyTorch 逐块实现/.test(courseSource)) {
    addError("assets/course.js still renders a separate PyTorch implementation section");
  }
  if (!/prism-python(?:\.min)?\.js/.test(chapterHtml)) {
    addError("chapter.html must load pinned Prism Python highlighting");
  }
  if (/<style\b/i.test(chapterHtml)) {
    addError("chapter.html must keep workbench styling in assets/styles.css, not inline");
  }
}

function splitLinesKeepEnds(source) {
  const lines = source.match(/[^\r\n]*(?:\r\n|\n|\r|$)/g) || [];
  if (lines.at(-1) === "") lines.pop();
  return lines;
}

function extractTeachingBlocks(source, sourceName) {
  const lines = splitLinesKeepEnds(source);
  const blocks = [];
  let active = null;

  lines.forEach((line, zeroBasedIndex) => {
    const lineNumber = zeroBasedIndex + 1;
    const markerLine = line.replace(/(?:\r\n|\n|\r)$/, "");
    const opening = markerLine.match(OPEN_MARKER);
    const closing = markerLine.match(CLOSE_MARKER);

    if (opening) {
      if (active) {
        addError(
          `${sourceName}:${lineNumber} opens Block ${opening[1]} inside ` +
            `unclosed Block ${active.id}`
        );
        return;
      }
      const expectedId = String(blocks.length + 1).padStart(2, "0");
      if (opening[1] !== expectedId) {
        addError(
          `${sourceName}:${lineNumber} expected Block ${expectedId}; found ${opening[1]}`
        );
      }
      active = {
        id: opening[1],
        title: opening[2],
        openingLine: lineNumber
      };
      return;
    }

    if (closing) {
      if (!active) {
        addError(
          `${sourceName}:${lineNumber} closes Block ${closing[1]} without an opening marker`
        );
        return;
      }
      if (closing[1] !== active.id) {
        addError(
          `${sourceName}:${lineNumber} closes Block ${active.id} with ` +
            `Block ${closing[1]}`
        );
      }
      const start = active.openingLine + 1;
      const end = lineNumber - 1;
      if (start > end) {
        addError(`${sourceName}:${active.openingLine} Block ${active.id} is empty`);
      }
      blocks.push({
        id: active.id,
        title: active.title,
        code: lines.slice(active.openingLine, lineNumber - 1).join(""),
        start,
        end
      });
      active = null;
      return;
    }

    if (MARKER_FRAGMENT.test(markerLine)) {
      addError(`${sourceName}:${lineNumber} has a malformed teaching-block marker`);
    }
  });

  if (active) {
    addError(
      `${sourceName}:${active.openingLine} Block ${active.id} has no closing marker`
    );
  }
  if (blocks.length === 0) {
    addError(`${sourceName} contains no teaching blocks`);
  }
  return blocks;
}

function validateImplementations(implementations, chapterIds) {
  if (
    !implementations ||
    typeof implementations !== "object" ||
    Array.isArray(implementations)
  ) {
    addError(
      "assets/implementations.js must assign an object to " +
        "window.ATTENTION_IMPLEMENTATIONS"
    );
    return;
  }

  const implementationIds = Object.keys(implementations);
  compareOrderedIds(implementationIds, chapterIds, "implementation ids");
  counts.implementations = implementationIds.length;

  implementationIds.forEach((chapterId) => {
    const implementation = implementations[chapterId];
    const location = `implementation ${chapterId}`;
    if (!implementation || typeof implementation !== "object") {
      addError(`${location} must be an object`);
      return;
    }

    const expectedPath = EXPECTED_SOURCE_PATHS.get(chapterId);
    if (implementation.path !== expectedPath) {
      addError(
        `${location}.path must be ${JSON.stringify(expectedPath)}; ` +
          `found ${JSON.stringify(implementation.path)}`
      );
    }
    if (!isNonEmptyString(implementation.source)) {
      addError(`${location}.source must be nonempty`);
      return;
    }

    const resolvedPath = path.resolve(ROOT, implementation.path || "");
    if (!resolvedPath.startsWith(`${ROOT}${path.sep}`)) {
      addError(`${location}.path must stay within the project root`);
      return;
    }

    let diskSource;
    try {
      diskSource = fs.readFileSync(resolvedPath, "utf8");
    } catch (error) {
      addError(`${location}.path cannot be read: ${error.message}`);
      return;
    }
    if (implementation.source !== diskSource) {
      addError(
        `${location}.source differs from ${implementation.path}; ` +
          "run `python3 tools/sync_pytorch_examples.py`"
      );
    }

    const extracted = extractTeachingBlocks(diskSource, implementation.path);
    if (!Array.isArray(implementation.blocks)) {
      addError(`${location}.blocks must be an array`);
      return;
    }
    counts.blocks += implementation.blocks.length;
    if (implementation.blocks.length !== extracted.length) {
      addError(
        `${location}.blocks has ${implementation.blocks.length} entries; ` +
          `${implementation.path} has ${extracted.length} marker pairs`
      );
    }

    const longest = Math.max(implementation.blocks.length, extracted.length);
    for (let index = 0; index < longest; index += 1) {
      const bundled = implementation.blocks[index];
      const actual = extracted[index];
      if (!bundled || !actual) continue;
      const blockLocation = `${location}.blocks[${index}]`;
      for (const field of ["id", "title", "start", "end", "code"]) {
        if (bundled[field] !== actual[field]) {
          const guidance =
            field === "code" || field === "start" || field === "end"
              ? " (regenerate with `python3 tools/sync_pytorch_examples.py`)"
              : "";
          addError(
            `${blockLocation}.${field} does not match marker-derived value${guidance}`
          );
        }
      }
    }
  });
}

function validateIndexHtml(indexHtml, chapters) {
  const chapterIds = new Set(chapters.map((chapter) => chapter.id));
  const linkedIds = new Set();
  const hrefPattern = /\bhref\s*=\s*(?:"([^"]*)"|'([^']*)')/gi;
  let match;

  while ((match = hrefPattern.exec(indexHtml)) !== null) {
    const href = match[1] ?? match[2];
    let parsed;
    try {
      parsed = new URL(href, "https://attention-atlas.invalid/");
    } catch {
      continue;
    }
    if (!parsed.pathname.endsWith("/chapter.html")) continue;
    counts.indexLinks += 1;
    const id = parsed.searchParams.get("id");
    if (!id) {
      addError(`index.html chapter link is missing an id: ${href}`);
    } else if (!chapterIds.has(id)) {
      addError(`index.html chapter link refers to unknown id ${JSON.stringify(id)}: ${href}`);
    } else {
      linkedIds.add(id);
    }
  }

  if (counts.indexLinks === 0) {
    addError("index.html must contain chapter.html?id=… links");
  }
  EXPECTED_IDS.forEach((id) => {
    if (!linkedIds.has(id)) {
      addError(`index.html has no hardcoded chapter link for ${JSON.stringify(id)}`);
    }
  });

  const metrics = new Map();
  const metricPattern =
    /<div\b[^>]*class=["'][^"']*\bmetric\b[^"']*["'][^>]*>\s*<strong([^>]*)>([^<]*)<\/strong>\s*<span>([^<]*)<\/span>\s*<\/div>/gi;
  while ((match = metricPattern.exec(indexHtml)) !== null) {
    const attributes = match[1];
    const value = match[2].trim().replace(/\s+/g, " ");
    const label = match[3].trim().replace(/\s+/g, " ");
    if (metrics.has(label)) {
      addError(`index.html metric label is duplicated: ${label}`);
    }
    metrics.set(label, { attributes, value });
  }

  const expectedMetrics = new Map([
    ["核心章节", String(counts.chapters)],
    ["演化分支", "3"],
    ["推导步骤 + 深度练习", String(counts.derivations + counts.exercises)],
    ["本地学习进度", `0 / ${counts.chapters}`]
  ]);
  expectedMetrics.forEach((expectedValue, label) => {
    const metric = metrics.get(label);
    if (!metric) {
      addError(`index.html is missing the ${JSON.stringify(label)} metric`);
    } else if (metric.value !== expectedValue) {
      addError(
        `index.html metric ${JSON.stringify(label)} must be ${JSON.stringify(
          expectedValue
        )}; found ${JSON.stringify(metric.value)}`
      );
    }
  });
  const progress = metrics.get("本地学习进度");
  if (progress && !/\bdata-progress-count\b/.test(progress.attributes)) {
    addError("index.html local progress metric must keep the data-progress-count hook");
  }
}

function validateNoLegacyDiagramMarkup(diagramSource) {
  for (const helper of ["subScript", "superScript", "svgLabel"]) {
    if (new RegExp(`\\b${helper}\\b`).test(diagramSource)) {
      addError(
        `assets/diagrams.js still contains legacy ${helper}; use mathLabel/KaTeX markup`
      );
    }
  }
  const unicodeSubscripts = diagramSource.match(/[\u2080-\u209f]/gu) || [];
  if (unicodeSubscripts.length > 0) {
    const unique = Array.from(new Set(unicodeSubscripts)).join(" ");
    addError(
      `assets/diagrams.js contains Unicode subscript emulation (${unique}); ` +
        "use TeX math labels"
    );
  }
  if (/baseline-shift\s*=\s*["'](?:sub|super)["']/i.test(diagramSource)) {
    addError(
      "assets/diagrams.js contains legacy SVG tspan sub/superscript emulation; " +
        "use TeX math labels"
    );
  }
}

function validateGeneratedBundleSync() {
  const generator = path.join(ROOT, "tools", "sync_pytorch_examples.py");
  const result = spawnSync("python3", [generator, "--check"], {
    cwd: ROOT,
    encoding: "utf8",
    timeout: 30_000
  });

  if (result.error) {
    addError(`could not check generated implementations bundle: ${result.error.message}`);
    return;
  }
  if (result.status !== 0) {
    const detail = `${result.stderr || result.stdout || ""}`.trim();
    addError(
      "generated implementations bundle is stale or invalid; run " +
        "`python3 tools/sync_pytorch_examples.py`" +
        (detail ? ` (${detail})` : "")
    );
  }
}

function main() {
  const context = createBrowserContext();
  executeBrowserAsset(context, "assets/chapters.js");
  const diagramSource = executeBrowserAsset(context, "assets/diagrams.js");
  executeBrowserAsset(context, "assets/implementations.js");

  const chapters = context.window.ATTENTION_CHAPTERS;
  const implementations = context.window.ATTENTION_IMPLEMENTATIONS;
  validateChapters(chapters);
  validateNoLegacyDiagramMarkup(diagramSource);
  validateDiagrams(chapters, context.window.AttentionDiagrams, implementations);

  const chapterIds = Array.isArray(chapters)
    ? chapters.map((chapter) => chapter.id)
    : EXPECTED_IDS;
  validateImplementations(
    implementations,
    chapterIds
  );
  const indexHtml = readText("index.html");
  const chapterHtml = readText("chapter.html");
  validateIndexHtml(indexHtml, Array.isArray(chapters) ? chapters : []);
  validateRenderer(readText("assets/course.js"), chapterHtml);
  validateGeneratedBundleSync();

  if (errors.length > 0) {
    console.error(`Content validation failed with ${errors.length} error(s):`);
    errors.forEach((error) => console.error(`- ${error}`));
    process.exitCode = 1;
    return;
  }

  console.log(
    "Content valid: " +
      `${counts.chapters} chapters, ${counts.derivations} derivations, ` +
      `${counts.exercises} exercises, ${counts.diagrams} diagrams, ` +
      `${counts.interactiveNodes} interactive nodes, ` +
      `${counts.implementations} implementations/${counts.blocks} blocks, ` +
      `${counts.indexLinks} index links; generated bundle synchronized.`
  );
}

try {
  main();
} catch (error) {
  console.error(`Content validation could not run: ${error.stack || error.message}`);
  process.exitCode = 1;
}
