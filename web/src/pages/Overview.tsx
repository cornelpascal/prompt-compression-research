import { Link } from "react-router-dom";
import {
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  FlaskConical,
  Layers,
  ShieldCheck,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Badge, Notice, SectionTitle } from "../components";
import { date, fmt } from "../data";
import type { Catalog } from "../types";

export default function Overview({ data }: { data: Catalog }) {
  const stages = data.paperkb.versions.map((v) => ({
    name: v.state.charAt(0).toUpperCase() + v.state.slice(1),
    count: v.count,
  }));
  return (
    <>
      <div className="hero">
        <div className="eyebrow">
          <span className="live-mark" />
          AN OPEN RESEARCH NOTEBOOK
        </div>
        <h1>
          Less context.
          <br />
          <em>More understanding.</em>
        </h1>
        <p>
          Investigating what a prompt can lose—and what it must preserve.
          Explore the literature, inspect the experiments, and follow the
          evidence.
        </p>
        <div className="hero-links">
          <Link className="button primary" to="/results">
            Explore the results
            <ArrowRight size={16} />
          </Link>
          <Link className="button subtle" to="/methods">
            Understand the methods
            <ArrowUpRight size={15} />
          </Link>
        </div>
        <div className="hero-byline">
          Research by <strong>Ganea Corneliu</strong>
          <span>·</span>Reproducible by design
        </div>
        <div
          className="hero-visual"
          aria-label="Conceptual diagram: source prompt to a shorter prompt with retained evidence"
        >
          <div className="visual-top">
            <span>INPUT CONTEXT</span>
            <span>→</span>
            <span>RETAINED SIGNAL</span>
          </div>
          <div className="token-diagram">
            <div className="token-source">
              {Array.from({ length: 36 }, (_, i) => (
                <i
                  key={i}
                  className={
                    [2, 3, 7, 12, 15, 18, 19, 25, 29, 32].includes(i)
                      ? "kept"
                      : ""
                  }
                  style={{ width: `${[29, 44, 35, 52, 24, 41][i % 6]}px` }}
                />
              ))}
            </div>
            <div className="compression-gate">
              <Layers size={26} />
              <small>SELECT · REWRITE</small>
            </div>
            <div className="token-dest">
              {Array.from({ length: 10 }, (_, i) => (
                <i key={i} style={{ width: `${[33, 48, 29, 41][i % 4]}px` }} />
              ))}
            </div>
          </div>
          <div className="visual-caption">
            <span>Preserve the task.</span>
            <span>Measure what changes.</span>
          </div>
          <div className="diagram-note">
            Conceptual illustration · not an experimental result
          </div>
        </div>
      </div>
      <div className="evidence-ribbon">
        <ShieldCheck size={18} />
        <strong>Evidence, with its boundaries.</strong>
        <span>
          Current compressor results use tiny-model fixtures. Real-model GPU
          studies are pending.
        </span>
        <Link to="/validation">
          See validation
          <ArrowRight size={14} />
        </Link>
      </div>
      <div className="stat-grid">
        <div>
          <span>PAPERS IN THE SNAPSHOT</span>
          <strong>{fmt(data.paperkb.papers)}</strong>
          <small>Literature and admitted predecessors</small>
        </div>
        <div>
          <span>REVIEWED IDEA RECORDS</span>
          <strong>{fmt(data.paperkb.reviewedIdeas)}</strong>
          <small>Source-linked, independently reviewed</small>
        </div>
        <div>
          <span>COMPRESSION APPROACHES</span>
          <strong>03</strong>
          <small>Rewrite · delete · attention</small>
        </div>
        <div>
          <span>RELATIVE F1 TARGET</span>
          <strong>
            98<span>%</span>
          </strong>
          <small>Acceptance criterion · not yet established</small>
        </div>
      </div>
      <SectionTitle
        number="01"
        title="Three connected research areas"
        aside={
          <span className="subtle-text">From evidence to experiments</span>
        }
      />
      <div className="research-grid">
        {[
          {
            n: "01",
            icon: BookOpen,
            title: "The evidence base",
            tag: "PAPERKB",
            text: "Trace research ideas to their sources. Inspect coverage, independent reviews, and the relationships that are actually supported.",
            link: "/literature",
            cta: "Explore the literature",
            kind: "historical_snapshot" as const,
          },
          {
            n: "02",
            icon: Layers,
            title: "Prompt compression",
            tag: "COMPRESSION LAB",
            text: "Compare generative rewriting, source-word deletion, and first-layer attention. Examine quality, budgets, and what gets removed.",
            link: "/methods",
            cta: "Explore the methods",
            kind: "engineering_fixture" as const,
          },
          {
            n: "03",
            icon: FlaskConical,
            title: "Privacy & utility",
            tag: "CONTROLLED LEAKAGE",
            text: "Study what compressed context still reveals. Separate useful task information from recoverable private information.",
            link: "/results?view=privacy",
            cta: "Inspect the tradeoff",
            kind: "synthetic_pilot" as const,
          },
        ].map((c) => (
          <Link to={c.link} key={c.n} className="research-card">
            <div className="card-index">
              <c.icon size={22} />
              <span>{c.n}</span>
            </div>
            <div className="eyebrow">{c.tag}</div>
            <h3>{c.title}</h3>
            <p>{c.text}</p>
            <Badge kind={c.kind} />
            <div className="card-cta">
              {c.cta}
              <ArrowUpRight size={17} />
            </div>
          </Link>
        ))}
      </div>
      <SectionTitle
        number="02"
        title="Where the research stands"
        aside={
          <Link className="text-link" to="/experiments">
            Open experiment notebook
            <ArrowRight size={14} />
          </Link>
        }
      />
      <div className="overview-bottom">
        <section className="panel">
          <div className="panel-heading">
            <h3>Evidence pipeline</h3>
            <Badge kind="historical_snapshot" />
          </div>
          <p className="muted">
            PDF versions by processing stage · {date(data.paperkb.recordedAt)}
          </p>
          <div className="small-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={stages}
                layout="vertical"
                margin={{ top: 10, right: 25, left: 15, bottom: 0 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  horizontal={false}
                  stroke="#e8e8e0"
                />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis
                  dataKey="name"
                  type="category"
                  tick={{ fontSize: 12 }}
                  width={75}
                />
                <Tooltip />
                <Bar
                  dataKey="count"
                  name="PDF versions"
                  fill="#376453"
                  radius={[0, 4, 4, 0]}
                  maxBarSize={24}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="caption">
            Prepared text is not completed review. {data.paperkb.complete}{" "}
            papers are complete across admitted versions. No production ancestry
            links are published.
          </p>
        </section>
        <section className="panel roadmap">
          <div className="panel-heading">
            <h3>Research checkpoints</h3>
            <span className="eyebrow">STATUS</span>
          </div>
          {[
            [
              "Implemented",
              "Training, evaluation & provenance",
              "Three research packages with reproducible commands.",
              "done",
            ],
            [
              "Verified on CPU",
              "Engineering and synthetic controls",
              "Numerical checks, recovery, inversion, and finite controls.",
              "done",
            ],
            [
              "Next experiment",
              "Real-model quality & GPU timing",
              "Test the 98% criterion and actual end-to-end speed.",
              "next",
            ],
          ].map(([tag, title, text, state]) => (
            <div className={"checkpoint " + state} key={tag}>
              <div className="checkpoint-dot" />
              <div>
                <span>{tag}</span>
                <h4>{title}</h4>
                <p>{text}</p>
              </div>
            </div>
          ))}
        </section>
      </div>
      <Notice>
        PaperKB data is a dated snapshot, not a live database view. Result
        labels distinguish engineering fixtures, synthetic measurements, and
        finite-source calculations throughout this notebook.
      </Notice>
    </>
  );
}
