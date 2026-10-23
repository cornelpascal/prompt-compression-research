import { useRef, useState, type ReactNode } from "react";
import {
  ArrowUpRight,
  Check,
  Copy,
  Download,
  Info,
  Table2,
} from "lucide-react";
import { csv, download, sourceUrl } from "./data";
import type { Evidence } from "./types";

export function Badge({
  kind,
  children,
}: {
  kind?: Evidence | "pending" | "success" | "neutral";
  children?: ReactNode;
}) {
  const text = {
    engineering_fixture: "Engineering fixture",
    synthetic_pilot: "Synthetic pilot",
    historical_snapshot: "Historical snapshot",
    finite_source: "Finite-source calculation",
    measured: "Measured result",
    pending: "Pending validation",
    success: "Verified",
    neutral: "Recorded",
  };
  return (
    <span className={`badge ${kind || "neutral"}`}>
      <span className="badge-dot" />
      {children || text[kind || "neutral"]}
    </span>
  );
}
export function PageTitle({
  eyebrow,
  title,
  description,
  children,
}: {
  eyebrow: string;
  title: string;
  description: string;
  children?: ReactNode;
}) {
  return (
    <div className="page-title">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {children}
    </div>
  );
}
export function SectionTitle({
  number,
  title,
  aside,
}: {
  number?: string;
  title: string;
  aside?: ReactNode;
}) {
  return (
    <div className="section-title">
      <h2>
        {number && <span>{number}</span>}
        {title}
      </h2>
      {aside}
    </div>
  );
}
export function Notice({ children }: { children: ReactNode }) {
  return (
    <div className="notice">
      <Info size={17} />
      <div>{children}</div>
    </div>
  );
}
export function Source({
  path,
  children = "View implementation",
}: {
  path: string;
  children?: ReactNode;
}) {
  return (
    <a
      className="text-link"
      href={sourceUrl(path)}
      target="_blank"
      rel="noreferrer"
    >
      {children}
      <ArrowUpRight size={14} />
    </a>
  );
}
export function Code({ children }: { children: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="code-block">
      <pre>{children}</pre>
      <button
        aria-label="Copy command"
        onClick={async () => {
          await navigator.clipboard.writeText(children);
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        }}
      >
        {copied ? <Check size={15} /> : <Copy size={15} />}
      </button>
    </div>
  );
}
export function Figure({
  title,
  subtitle,
  evidence,
  rows,
  children,
  observations,
  interpretation,
  limitations,
  reproduce,
}: {
  title: string;
  subtitle: string;
  evidence: Evidence;
  rows: Record<string, unknown>[];
  children: ReactNode;
  observations: string;
  interpretation: string;
  limitations: string;
  reproduce: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [table, setTable] = useState(false);
  const keys = [...new Set(rows.flatMap((r) => Object.keys(r)))].filter(
    (k) => !["err"].includes(k),
  );
  const filename = title.toLowerCase().replace(/[^a-z0-9]+/g, "-");
  function svgExport() {
    const svgs = Array.from(
      ref.current?.querySelectorAll<SVGSVGElement>("svg.recharts-surface") ||
        [],
    );
    const svg = svgs.sort(
      (a, b) =>
        b.getBoundingClientRect().width - a.getBoundingClientRect().width,
    )[0];
    if (!svg) return;
    const width = svg.getBoundingClientRect().width,
      height = svg.getBoundingClientRect().height;
    const ns = "http://www.w3.org/2000/svg",
      output = document.createElementNS(ns, "svg");
    output.setAttribute("xmlns", ns);
    output.setAttribute("width", String(width));
    output.setAttribute("height", String(height + 110));
    output.setAttribute("viewBox", `0 0 ${width} ${height + 110}`);
    const background = document.createElementNS(ns, "rect");
    background.setAttribute("width", "100%");
    background.setAttribute("height", "100%");
    background.setAttribute("fill", "#fcfcf8");
    output.append(background);
    const legend = Array.from(
      ref.current?.querySelectorAll(".recharts-legend-item-text") || [],
    )
      .map((n) => n.textContent)
      .join(" · ");
    for (const [index, text] of [
      title,
      `${evidence.replaceAll("_", " ")} · Ganea Corneliu`,
      legend,
    ].entries()) {
      const line = document.createElementNS(ns, "text");
      line.setAttribute("x", "16");
      line.setAttribute("y", String(24 + index * 23));
      line.setAttribute("font-size", index === 0 ? "15" : "11");
      line.setAttribute("font-family", "sans-serif");
      line.setAttribute("fill", "#394b32");
      line.textContent = text;
      output.append(line);
    }
    const clone = svg.cloneNode(true) as SVGElement;
    clone.setAttribute("y", "85");
    output.append(clone);
    download(
      filename + ".svg",
      new XMLSerializer().serializeToString(output),
      "image/svg+xml",
    );
  }
  return (
    <section className="figure" ref={ref}>
      <div className="figure-header">
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
        <Badge kind={evidence} />
      </div>
      <div className="figure-actions">
        <button
          className={table ? "active" : ""}
          onClick={() => setTable(!table)}
        >
          <Table2 size={14} />
          {table ? "Show chart" : "Data table"}
        </button>
        <button
          onClick={() => download(filename + ".csv", csv(rows), "text/csv")}
        >
          <Download size={14} />
          CSV
        </button>
        <button onClick={svgExport} disabled={table}>
          <Download size={14} />
          SVG
        </button>
      </div>
      {table ? (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                {keys.map((k) => (
                  <th key={k}>{k.replaceAll("_", " ")}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={i}>
                  {keys.map((k) => (
                    <td key={k}>
                      {typeof row[k] === "number"
                        ? Number(row[k]).toFixed(3)
                        : String(row[k] ?? "Not measured")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="chart-area">{children}</div>
      )}
      <div className="figure-analysis">
        <div>
          <span>Observation</span>
          <p>{observations}</p>
        </div>
        <div>
          <span>Interpretation</span>
          <p>{interpretation}</p>
        </div>
        <div>
          <span>Limits of this evidence</span>
          <p>{limitations}</p>
        </div>
      </div>
      <details className="reproduce-figure">
        <summary>Reproduce this figure</summary>
        <Code>{reproduce}</Code>
      </details>
    </section>
  );
}
export function Empty({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}

export function Filter({
  name,
  value,
  onChange,
  options,
}: {
  name: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <label className="filter">
      <span>{name}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}
