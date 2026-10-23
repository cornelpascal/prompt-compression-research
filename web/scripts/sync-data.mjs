import { cp, mkdir } from "node:fs/promises";
await mkdir(new URL("../public/data/", import.meta.url), { recursive: true });
await cp(
  new URL("../../site-data/", import.meta.url),
  new URL("../public/data/", import.meta.url),
  { recursive: true },
);
