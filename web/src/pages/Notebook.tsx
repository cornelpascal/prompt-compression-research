import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  ArrowRight,
  ArrowUpRight,
  CheckCircle2,
  CircleDashed,
  Download,
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Badge,
  Code,
  Figure,
  Notice,
  PageTitle,
  SectionTitle,
  Source,
  Filter,
} from "../components";
import { date, download, fmt, label, pct, resource } from "../data";
import { metrics, testGroups } from "../content";
import type { Catalog } from "../types";

export function Datasets({ data }: { data: Catalog }) {
  const squad = data.datasets[0];
  return (
    <>
      <PageTitle
        eyebrow="DATASETS & EXPERIMENTAL PROTOCOLS"
        title="Define the experiment first."
        description="A result is only interpretable when its data, units, inference boundary, and split policy are explicit."
      />
      <section className="panel prose">
        <div className="panel-heading">
          <h2>SQuAD 1.1</h2>
          <a
            className="text-link"
            href={squad.url}
            target="_blank"
            rel="noreferrer"
          >
            Dataset source
            <ArrowUpRight size={14} />
          </a>
        </div>
        <p>{squad.note}</p>
        <div className="split-bar">
          {[
            ["Training", squad.train, "train"],
            ["Validation", squad.validation, "validation"],
            ["Held-out test", squad.test, "test"],
          ].map(([name, count, id]) => (
            <div
              key={id}
              className={String(id)}
              style={{ flex: Number(count) }}
              title={`${name}: ${count}`}
            >
              <span>{name}</span>
            </div>
          ))}
        </div>
        <div className="stat-grid compact">
          {[
            ["Training", squad.train, squad.groups[0]],
            ["Validation", squad.validation, squad.groups[1]],
            ["Held-out test", squad.test, squad.groups[2]],
          ].map(([name, count, groups]) => (
            <div key={String(name)}>
              <span>{name}</span>
              <strong>{fmt(Number(count))}</strong>
              <small>{groups} articles</small>
            </div>
          ))}
        </div>
        <p className="caption">
          License: {squad.license}. Article grouping prevents the same article
          from crossing training and validation. Gold answers are not part of
          the compressor inference API.
        </p>
      </section>
      <div className="two-columns">
        <section className="panel prose">
          <h3>Synthetic privacy records</h3>
          <p>
            Structured tasks cover scheduling, routing, and eligibility-style
            decisions. Private-code counterfactuals change protected values
            while preserving nonessential-task answers.
          </p>
          <p>
            Training, validation, test, and correlation-shift splits have
            disjoint entities and template identifiers. The shift reverses the
            group–outcome correlation.
          </p>
          <Badge kind="synthetic_pilot" />
          <p>
            The saved CPU pilot contains{" "}
            <strong>{fmt(data.privacy.rows)} evaluation rows</strong> and{" "}
            {data.privacy.summary.length} summary groups. Length is measured in
            UTF-8 bytes.
          </p>
          <Source path="research/controlled_leakage/docs/PROTOCOL.md" />
        </section>
        <section className="panel prose">
          <h3>Paper corpus</h3>
          <p>
            The historical snapshot includes 1,451 original corpus papers and
            114 admitted predecessors. Each PDF version has separate
            preparation, extraction, review, and embedding state.
          </p>
          <p>
            Original papers can nominate relevant predecessors. Added
            predecessors do not expand the corpus another hop. Production
            ancestry requires independently reviewed directional evidence and
            verified chronology.
          </p>
          <Badge kind="historical_snapshot" />
          <p>
            Thirteen withdrawn sources use recorded earlier-version
            substitutions. A source gap remains visible rather than being filled
            by an unrelated substitute.
          </p>
          <Source path="WORKER_PROTOCOL.md" />
        </section>
      </div>
      <SectionTitle title="Metric glossary" />
      <div className="glossary">
        {metrics.map(([name, text]) => (
          <div key={name}>
            <h3>{name}</h3>
            <p>{text}</p>
          </div>
        ))}
      </div>
      <Notice>
        The complete original prompt is used for the evaluation baseline. The
        compression receiver gets the compressed text alone; the question is not
        secretly supplied a second time.
      </Notice>
    </>
  );
}

export function Experiments({ data }: { data: Catalog }) {
  const [selected, setSelected] = useState(data.compression[0].id);
  const run = data.compression.find((r) => r.id === selected)!;
  const [trainingIndex, setTrainingIndex] = useState(0);
  const training =
    run.training[Math.min(trainingIndex, run.training.length - 1)];
  return (
    <>
      <PageTitle
        eyebrow="EXPERIMENT NOTEBOOK"
        title="Every run leaves a trail."
        description="Explore saved configurations, training traces, evaluation identities, and the precise boundary between an engineering check and a scientific result."
      />
      <div className="run-cards">
        {data.compression.map((r) => (
          <button
            key={r.id}
            className={"panel run-card " + (r.id === run.id ? "selected" : "")}
            onClick={() => {
              setSelected(r.id);
              setTrainingIndex(0);
            }}
          >
            <Badge kind={r.evidence} />
            <h3>{r.label}</h3>
            <p>
              {date(r.recordedAt)} · {r.cases.length} receiver-level rows
            </p>
            <span className="mono">{r.identity.slice(0, 14)}</span>
          </button>
        ))}
      </div>
      <section className="panel prose">
        <div className="panel-heading">
          <h2>{run.label}</h2>
          <Link className="text-link" to="/results">
            Inspect results
            <ArrowRight size={14} />
          </Link>
        </div>
        <div className="definition-grid">
          <div>
            <span>Profile</span>
            <strong>{run.profile}</strong>
          </div>
          <div>
            <span>Length unit</span>
            <strong>{run.unit}</strong>
          </div>
          <div>
            <span>Recorded date</span>
            <strong>{date(run.recordedAt)}</strong>
          </div>
          <div>
            <span>Methods in this run</span>
            <strong>
              {[...new Set(run.summary.map((s) => label(s.method)))].join(", ")}
            </strong>
          </div>
        </div>
        <details>
          <summary>Provenance and limitations</summary>
          <p>Evaluation identity</p>
          <code className="hash">{run.identity}</code>
          <p>Recorded source-code hash</p>
          <code className="hash">{run.sourceHash}</code>
          <ul>
            {run.limitations.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
          <button
            className="button subtle"
            onClick={() =>
              download(
                run.id + ".json",
                JSON.stringify(run, null, 2),
                "application/json",
              )
            }
          >
            <Download size={14} />
            Download this run
          </button>
        </details>
      </section>
      {training && (
        <>
          <Filter
            name="Training trace"
            value={String(Math.min(trainingIndex, run.training.length - 1))}
            onChange={(v) => setTrainingIndex(Number(v))}
            options={run.training.map((t, i) => ({
              value: String(i),
              label: `${label(t.method)} · ${t.stage.toUpperCase()}`,
            }))}
          />
          <Figure
            title="Training loss within this run"
            subtitle={`${label(training.method)} · ${training.stage.toUpperCase()} · ${training.steps} optimizer steps`}
            evidence="engineering_fixture"
            rows={training.losses.map((l) => ({
              step: l.step,
              loss: l.loss,
              gradient_norm: l.gradient_norm,
            }))}
            observations={`The saved trace contains ${training.losses.length} logged updates over ${training.steps} optimizer steps. Training took ${training.seconds.toFixed(2)} seconds in the recorded CPU environment.`}
            interpretation="This establishes that the optimization and checkpoint path executed. Inspect the loss within its own objective; different methods use different losses."
            limitations="A short tiny-model trace is not evidence of convergence, downstream improvement, or GPU throughput. CPU memory values of zero mean unmeasured."
            reproduce="python -m promptcomp.smoke --output runs/new-cpu-check\npython -m promptcomp.attention_smoke --output runs/new-attention-check"
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={training.losses}
                margin={{ top: 20, right: 30, bottom: 25, left: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e6df" />
                <XAxis
                  dataKey="step"
                  label={{
                    value: "Optimizer step",
                    position: "bottom",
                    offset: 8,
                  }}
                />
                <YAxis domain={["auto", "auto"]} />
                <Tooltip />
                <Line
                  type="linear"
                  dataKey="loss"
                  stroke="#285d4b"
                  strokeWidth={2}
                  dot={{ r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </Figure>
        </>
      )}
      <section className="panel prose">
        <h3>Comparability rules</h3>
        <p>
          Only compare runs with compatible datasets, splits, receiver
          configurations, measurement units, and evidence categories. A
          byte-based synthetic pilot is not directly comparable with a
          token-based SQuAD evaluation. The UI keeps these domains separate.
        </p>
        <p>
          Model revisions, source hashes, dataset hashes, checkpoint identities,
          and decoding settings are part of the reproducibility record. Resuming
          with changed inputs requires a new run directory.
        </p>
      </section>
    </>
  );
}

type LiveTestRecord = {
  id: string;
  task: string;
  status: string;
  recordedAt: string;
  url: string;
  suites?: {
    name: string;
    tests: number;
    failures: number;
    errors: number;
    skipped: number;
  }[];
};
function LatestTests() {
  const [record, setRecord] = useState<LiveTestRecord>();
  useEffect(() => {
    resource<LiveTestRecord[]>("latest-jobs.json")
      .then((rows) => setRecord(rows.find((r) => r.task === "tests")))
      .catch(() => {});
  }, []);
  if (!record) return null;
  return (
    <section className="panel prose">
      <div className="panel-heading">
        <h2>Latest isolated CI run</h2>
        <Badge kind={record.status === "success" ? "success" : "pending"}>
          {record.status}
        </Badge>
      </div>
      <p>
        {date(record.recordedAt)} · GitHub-hosted CPU with a disposable
        PostgreSQL database.
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Suite</th>
              <th>Passed</th>
              <th>Failures</th>
              <th>Errors</th>
              <th>Skipped</th>
            </tr>
          </thead>
          <tbody>
            {record.suites?.map((s) => (
              <tr key={s.name}>
                <td>{s.name}</td>
                <td>{s.tests - s.failures - s.errors - s.skipped}</td>
                <td>{s.failures}</td>
                <td>{s.errors}</td>
                <td>{s.skipped}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="caption">
        The two distributed tests are skipped in the default compression suite
        and run separately. These checks establish engineering behavior; they do
        not measure pretrained-model quality.
      </p>
      <a
        className="text-link"
        href={record.url}
        target="_blank"
        rel="noreferrer"
      >
        Inspect run and JUnit artifacts
        <ArrowUpRight size={14} />
      </a>
    </section>
  );
}

export function Validation({ data }: { data: Catalog }) {
  return (
    <>
      <PageTitle
        eyebrow="TESTS & VALIDATION"
        title="What has actually been checked."
        description="Recorded test outcomes, the properties they verify, and the experiments that remain. Test counts are reported by suite; overlapping historical runs are not added together."
      />
      <LatestTests />
      <SectionTitle title="Historical validation records" />
      <div className="test-records">
        {data.tests.map((t) => (
          <section className="test-record" key={t.id}>
            <div className="test-icon">
              {t.status.includes("blocked") ? (
                <CircleDashed size={23} />
              ) : (
                <CheckCircle2 size={23} />
              )}
            </div>
            <div className="test-description">
              <h3>{t.title}</h3>
              <p>{t.scope}</p>
              <small>
                {date(t.date)} · {t.status}
              </small>
            </div>
            <div className="test-count">
              <strong>{t.passed}</strong>
              <span>passed</span>
            </div>
            <div className="test-other">
              <span>{t.skipped ?? "—"} skipped</span>
              <span>
                {t.failed ?? "—"}{" "}
                {t.status.includes("blocked") ? "setup errors" : "failed"}
              </span>
            </div>
            <Source path={t.source}>Evidence</Source>
          </section>
        ))}
      </div>
      <Notice>
        The latest local PaperKB attempt could not reach PostgreSQL. Its
        historical successful run remains in the record. Fresh CI results use a
        disposable database and do not overwrite this history.
      </Notice>
      <SectionTitle title="What each test group establishes" />
      <div className="two-columns">
        {testGroups.map((g) => (
          <section className="panel prose" key={g.name}>
            <div className="eyebrow">{g.project}</div>
            <h3>{g.name}</h3>
            <h4>Checks</h4>
            <p>{g.checks}</p>
            <h4>Interpretation</h4>
            <p>{g.meaning}</p>
            <Source path={g.source}>Read the tests</Source>
          </section>
        ))}
      </div>
      <section className="panel prose">
        <h3>Still requiring research infrastructure</h3>
        <ul>
          <li>
            Real pretrained-model SQuAD quality and the 98% relative-F1
            threshold.
          </li>
          <li>
            Warmed, uncached GPU latency, peak memory, and end-to-end speedup.
          </li>
          <li>CUDA/NCCL and multi-node checkpoint recovery.</li>
          <li>
            Learned privacy-compressor comparisons against adaptive attacks.
          </li>
          <li>
            Complete corpus processing and a production ancestry graph that
            passes every barrier.
          </li>
        </ul>
        <Link className="button primary" to="/run">
          Run a bounded check
          <ArrowRight size={16} />
        </Link>
      </section>
    </>
  );
}

export function Reproduce() {
  return (
    <>
      <PageTitle
        eyebrow="REPRODUCIBILITY"
        title="From a figure to a command."
        description="The public repository includes the implementation, configurations, tests, and a curated evidence snapshot. Large datasets, model weights, database contents, and credentials stay outside the repository."
      />
      <section className="panel prose">
        <h2>Start with the CPU checks</h2>
        <p>
          Python 3.12 is the reference runtime for the research packages. The UI
          uses Node.js 24 and a committed npm lockfile.
        </p>
        <Code>
          {
            "git clone https://github.com/cornelpascal/prompt-compression-research.git\ncd prompt-compression-research\npython3.12 -m venv .venv\nsource .venv/bin/activate\npython -m pip install -r requirements-ci.txt\npython -m pip install --no-deps -e . -e research/prompt_compression -e research/controlled_leakage"
          }
        </Code>
        <Code>
          {
            "python -m pytest -q research/prompt_compression/tests research/controlled_leakage/tests\npython -m promptcomp.attention_smoke --output verification/attention-demo\ncd web && npm ci && npm run dev"
          }
        </Code>
      </section>
      <div className="two-columns">
        <section className="panel prose">
          <h3>Reproduce the full studies</h3>
          <p>
            The engineering demos validate execution using tiny random weights.
            Full studies require downloaded models, prepared data, and
            appropriate hardware.
          </p>
          <div className="link-list">
            <Source path="research/prompt_compression/README.md">
              Rewrite and deletion study
            </Source>
            <Source path="research/prompt_compression/ATTENTION.md">
              Attention study and calibration
            </Source>
            <Source path="research/controlled_leakage/README.md">
              Privacy and inversion experiments
            </Source>
            <Source path="PAPERKB.md">PaperKB setup and source workflow</Source>
          </div>
        </section>
        <section className="panel prose">
          <h3>Update the public evidence</h3>
          <p>
            The snapshot exporter reads the original workspace and produces
            allowlisted, checksum-validated JSON. It does not publish raw
            exports, leases, credentials, or PDF pages.
          </p>
          <Code>
            {
              "python scripts/export_site.py --workspace /path/to/research-workspace\npython scripts/validate_site_data.py\ncd web && npm run build"
            }
          </Code>
          <p>
            Use the run center for bounded CPU workflows. Successful jobs retain
            result artifacts and publish a normalized outcome; a failed test
            remains visible as a failure.
          </p>
          <Source path="docs/RUNNERS.md">Runner and workflow setup</Source>
        </section>
      </div>
      <section className="panel prose">
        <h3>Read a result responsibly</h3>
        <ol>
          <li>Check the evidence category and measurement date.</li>
          <li>
            Check the dataset, split, receiver, seed, units, and achieved
            budget.
          </li>
          <li>
            Inspect the uncertainty and compare only compatible conditions.
          </li>
          <li>Read the case-level examples and documented failure modes.</li>
          <li>
            Reproduce the command with the same pinned inputs before extending a
            claim.
          </li>
        </ol>
      </section>
    </>
  );
}

export function About({ data }: { data: Catalog }) {
  return (
    <>
      <PageTitle
        eyebrow="ABOUT THIS RESEARCH"
        title="Ganea Corneliu"
        description="Author of this research workspace and its public notebook on prompt compression, source-grounded evidence, and controlled information leakage."
      />
      <div className="two-columns">
        <section className="panel prose">
          <h2>The research question</h2>
          <p>
            How much of a prompt can be removed while preserving the task—and
            what information remains recoverable after compression?
          </p>
          <p>
            This workspace connects three complementary activities: inspecting
            the literature, building and evaluating compressors, and studying
            privacy–utility tradeoffs under explicit assumptions.
          </p>
          <p>
            The site is an inspectable research record. Implemented components,
            engineering checks, and established scientific findings are
            deliberately distinguished.
          </p>
          <a
            className="text-link"
            href="https://github.com/cornelpascal"
            target="_blank"
            rel="noreferrer"
          >
            GitHub · cornelpascal
            <ArrowUpRight size={14} />
          </a>
        </section>
        <section className="panel prose">
          <h2>Attribution and citation</h2>
          <p>
            Site and research workspace: <strong>Ganea Corneliu</strong>.
            Referenced papers, datasets, and models retain their original
            authorship and licenses.
          </p>
          <Code>
            {
              "Ganea Corneliu. Prompt Compression Research.\nResearch software and evidence notebook, 2026.\nhttps://cornelpascal.github.io/prompt-compression-research/"
            }
          </Code>
          <p>
            SQuAD 1.1 is attributed to its original authors and distributed
            under CC BY-SA 4.0. Literature entries link to their original
            sources; corpus inclusion is not an endorsement.
          </p>
        </section>
      </div>
      <SectionTitle title="Evidence provenance" />
      <div className="table-scroll panel">
        <table>
          <thead>
            <tr>
              <th>Source</th>
              <th>Recorded</th>
              <th>SHA-256</th>
            </tr>
          </thead>
          <tbody>
            {data.sources.map((s) => (
              <tr key={s.name}>
                <td>{s.name}</td>
                <td>{date(s.date)}</td>
                <td className="hash">{s.sha256}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="caption">
        Site snapshot generated {date(data.generatedAt)}. Source dates are
        preserved individually; a new website build does not make old
        observations current.
      </p>
      <button
        className="button subtle"
        onClick={() =>
          download(
            "research-catalog.json",
            JSON.stringify(data, null, 2),
            "application/json",
          )
        }
      >
        <Download size={15} />
        Download the curated research catalog
      </button>
    </>
  );
}
