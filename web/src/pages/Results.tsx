import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  CartesianGrid,
  ErrorBar,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Badge,
  Figure,
  Notice,
  PageTitle,
  SectionTitle,
  Filter,
} from "../components";
import { label, pct, privacySeries } from "../data";
import type { Catalog, Case } from "../types";

const colors = [
  "#285d4b",
  "#bd6b4d",
  "#7b79aa",
  "#779333",
  "#357f9b",
  "#9b655e",
  "#676c79",
];
const unique = (values: string[]) => [...new Set(values)].sort();

function Highlight({ item }: { item: Case }) {
  if (!item.example) return <p>Source example unavailable in this export.</p>;
  const prompt = item.example.prompt;
  const spans = item.compression.spans || [];
  let cursor = 0;
  const nodes = [];
  if (!spans.length) return <pre className="prompt-text">{prompt}</pre>;
  spans.forEach(([start, end], i) => {
    if (start > cursor)
      nodes.push(
        <span className="removed" key={"r" + i}>
          {prompt.slice(cursor, start)}
        </span>,
      );
    nodes.push(<mark key={i}>{prompt.slice(start, end)}</mark>);
    cursor = end;
  });
  if (cursor < prompt.length)
    nodes.push(
      <span className="removed" key="end">
        {prompt.slice(cursor)}
      </span>,
    );
  return <pre className="prompt-text">{nodes}</pre>;
}

export default function Results({ data }: { data: Catalog }) {
  const [params, setParams] = useSearchParams();
  const view = params.get("view") || "compression";
  return (
    <>
      <PageTitle
        eyebrow="EXPERIMENTAL EVIDENCE"
        title="Results, with context."
        description="Inspect the measurements and the conditions behind them. These views keep engineering demonstrations, synthetic experiments, and finite-source calculations separate."
      />
      <div className="tabs" role="tablist" aria-label="Result domain">
        {[
          ["compression", "Prompt compression"],
          ["privacy", "Privacy & utility"],
          ["theory", "Finite-source frontier"],
        ].map(([id, name]) => (
          <button
            role="tab"
            aria-selected={view === id}
            className={view === id ? "selected" : ""}
            key={id}
            onClick={() => setParams({ view: id })}
          >
            {name}
          </button>
        ))}
      </div>
      {view === "privacy" ? (
        <PrivacyResults data={data} />
      ) : view === "theory" ? (
        <TheoryResults data={data} />
      ) : (
        <CompressionResults data={data} />
      )}
    </>
  );
}

function CompressionResults({ data }: { data: Catalog }) {
  const [runId, setRunId] = useState("attention-validation");
  const [target, setTarget] = useState("qwen");
  const [caseIndex, setCaseIndex] = useState(0);
  const run =
    data.compression.find((r) => r.id === runId) || data.compression[0];
  const targets = unique(run.summary.map((s) => s.target));
  const effectiveTarget = targets.includes(target) ? target : targets[0];
  const records = run.summary.filter((s) => s.target === effectiveTarget);
  const names = unique(records.map((r) => r.method));
  const points = records.map((r) => ({
    method: label(r.method),
    key: r.method,
    retention: r.request_retained_fraction * 100,
    f1: r.f1.mean * 100,
    low: r.f1.low * 100,
    high: r.f1.high * 100,
    requested: r.requested_retained_fraction * 100,
    cases: r.cases,
    budget_compliance: r.budget_compliance * 100,
    err: [
      Math.max(0, (r.f1.mean - r.f1.low) * 100),
      Math.max(0, (r.f1.high - r.f1.mean) * 100),
    ],
  }));
  const cases = run.cases.filter((c) => c.target === effectiveTarget);
  const item = cases[Math.min(caseIndex, cases.length - 1)];
  return (
    <>
      <div className="filter-bar">
        <Filter
          name="Experiment"
          value={run.id}
          onChange={(v) => {
            setRunId(v);
            setCaseIndex(0);
          }}
          options={data.compression.map((r) => ({
            value: r.id,
            label: r.label,
          }))}
        />
        <Filter
          name="Receiver alias"
          value={effectiveTarget}
          onChange={(v) => {
            setTarget(v);
            setCaseIndex(0);
          }}
          options={targets.map((t) => ({
            value: t,
            label: t + " · fixture receiver",
          }))}
        />
        <Badge kind="engineering_fixture" />
      </div>
      <Notice>
        These runs use tiny randomly initialized models and artificial receiver
        logic. “Qwen”, “Mistral”, and “Phi” are fixture aliases here. The plots
        do not measure the corresponding pretrained models.
      </Notice>
      <Figure
        title="Answer quality versus retained context"
        subtitle="Receiver request tokens include the native chat wrapper. Lower retention is shorter; higher F1 is better."
        evidence="engineering_fixture"
        rows={points}
        observations={`${records.length} method–budget measurements are available for this fixture receiver. The selected run contains ${Math.max(...records.map((r) => r.cases))} cases per reported point.`}
        interpretation="The plot validates the measurement pipeline and exposes how outputs fit the budget. It does not establish a ranking of useful real-world compressors."
        limitations="Article-bootstrap intervals are shown where available. Tiny samples and artificial answers cannot support general accuracy claims. Different runs are not automatically comparable."
        reproduce="python -m promptcomp.attention_smoke --output runs/new-attention-check\npromptcomp report --run runs/new-attention-check/validation --output runs/new-attention-check/report"
      >
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 18, right: 32, bottom: 28, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e6df" />
            <XAxis
              type="number"
              dataKey="retention"
              name="Retained request"
              unit="%"
              domain={[0, 105]}
              label={{
                value: "Achieved receiver-token retention (%)",
                position: "bottom",
                offset: 10,
              }}
              tick={{ fontSize: 11 }}
            />
            <YAxis
              type="number"
              dataKey="f1"
              name="F1"
              unit="%"
              domain={[0, 105]}
              tick={{ fontSize: 11 }}
            />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} />
            <Legend verticalAlign="top" height={35} />
            {names.map((m, i) => (
              <Scatter
                name={label(m)}
                key={m}
                data={points
                  .filter((p) => p.key === m)
                  .sort((a, b) => a.retention - b.retention)}
                fill={colors[i % colors.length]}
                line={points.filter((p) => p.key === m).length > 1}
                shape="circle"
              >
                <ErrorBar
                  dataKey="err"
                  direction="y"
                  width={4}
                  stroke={colors[i % colors.length]}
                />
              </Scatter>
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </Figure>
      {run.threshold && (
        <>
          <SectionTitle title="The locked 98% acceptance check" />
          <p className="section-intro">
            Each operating point was chosen using validation only, then
            evaluated unchanged. A fixture pass is a software check, not
            real-model quality validation.
          </p>
          <div className="table-scroll panel">
            <table>
              <thead>
                <tr>
                  <th>Variant</th>
                  <th>Receiver</th>
                  <th>Retained budget</th>
                  <th>Relative F1</th>
                  <th>Margin over target</th>
                  <th>Fixture outcome</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(run.threshold.entries).flatMap(([key, e]) =>
                  Object.entries(e.targets).map(([t, v]) => (
                    <tr key={key + t}>
                      <td>{label(key.split(":")[0])}</td>
                      <td>{t} · fixture</td>
                      <td>{pct(e.ratio)}</td>
                      <td>{pct(v.relative_f1, 1)}</td>
                      <td>{(v.threshold_margin.mean * 100).toFixed(1)} pp</td>
                      <td>
                        <Badge kind={v.passes ? "success" : "pending"}>
                          {v.passes ? "Pass in fixture" : "Did not qualify"}
                        </Badge>
                      </td>
                    </tr>
                  )),
                )}
              </tbody>
            </table>
          </div>
        </>
      )}
      <SectionTitle
        title="Inspect an individual case"
        aside={<span className="subtle-text">The text behind the score</span>}
      />
      <Filter
        name="Case / method / requested retention"
        value={String(Math.min(caseIndex, cases.length - 1))}
        onChange={(v) => setCaseIndex(Number(v))}
        options={cases.map((c, i) => ({
          value: String(i),
          label: `${c.id} · ${label(c.method)} · ${pct(c.ratio)}`,
        }))}
      />
      {item && (
        <>
          <div className="case-metrics">
            <span>
              F1 <strong>{pct(item.score.f1)}</strong>
            </span>
            <span>
              Exact match <strong>{item.score.em ? "Yes" : "No"}</strong>
            </span>
            <span>
              Source → output{" "}
              <strong>
                {item.compression.input_tokens} →{" "}
                {item.compression.output_tokens} tokens
              </strong>
            </span>
            <span>
              Status <strong>{item.compression.status}</strong>
            </span>
          </div>
          <div className="two-columns">
            <section className="panel">
              <h3>Original prompt</h3>
              <p className="caption">
                Highlighted words are retained when extraction spans are
                available.
              </p>
              <Highlight item={item} />
            </section>
            <section className="panel">
              <h3>Compressed prompt</h3>
              <pre className="prompt-text">
                {item.compression.text || "(empty output)"}
              </pre>
              <div className="answer-box">
                <span>RECEIVER ANSWER</span>
                <strong>{item.answer.text}</strong>
                <small>
                  Reference:{" "}
                  {item.example?.answers.map((a) => a.text).join(" / ") ||
                    "Unavailable"}
                </small>
              </div>
              <p className="caption">
                {item.score.f1 === 1
                  ? "The fixture receiver matches the reference on this case. Its artificial decision rule is not evidence of semantic preservation."
                  : "The compressed text did not produce a matching answer under the fixture rule. Inspect the retained evidence and any budget failure before attributing a cause."}
              </p>
            </section>
          </div>
        </>
      )}
    </>
  );
}

function PrivacyResults({ data }: { data: Catalog }) {
  const rows = data.privacy.summary;
  const [regime, setRegime] = useState("correlated");
  const [domain, setDomain] = useState("scheduling");
  const [access, setAccess] = useState("context");
  const [attacker, setAttacker] = useState("attribute_nb");
  const selected = privacySeries(rows, regime, domain, access, attacker);
  const names = unique(selected.map((r) => r.method));
  const points = selected.map((r) => ({
    method: label(r.method),
    key: r.method,
    utility: r.utility * 100,
    recovery: r.secret_recovery * 100,
    retention: r.achieved_retention * 100,
    requested: r.ratio * 100,
    compliance: r.budget_compliance * 100,
    bytes: r.context_tokens,
  }));
  const comparisons = data.privacy.comparisons.filter(
    (r) =>
      r.regime === regime &&
      r.domain === domain &&
      r.access === access &&
      r.attacker === attacker,
  );
  return (
    <>
      <div className="filter-bar">
        {[
          ["Regime", regime, setRegime, "regime"],
          ["Domain", domain, setDomain, "domain"],
          ["Access", access, setAccess, "access"],
          ["Attacker", attacker, setAttacker, "attacker"],
        ].map(([name, value, set, key]) => (
          <Filter
            key={String(key)}
            name={String(name)}
            value={String(value)}
            onChange={set as (s: string) => void}
            options={unique(
              rows.map((r) => String(r[key as keyof typeof r])),
            ).map((v) => ({ value: v, label: label(v) }))}
          />
        ))}
      </div>
      <Figure
        title="Useful information, recoverable information"
        subtitle="Task utility versus literal secret recovery. The desirable direction is toward high utility and low recovery."
        evidence="synthetic_pilot"
        rows={points}
        observations={`${selected.length} control–budget cells are shown for ${domain}, ${regime} secrets, ${access} access, and the ${label(attacker)} attacker.`}
        interpretation="Compare recovery at similar task utility and achieved length. An empty context may hide a secret while failing the task; that is not a useful compression improvement."
        limitations="The source records and task checker are synthetic. Original and redaction-only controls can exceed the requested budget. These curves use UTF-8 bytes, not model tokens."
        reproduce="leaklab run --config research/controlled_leakage/configs/pilot.json --data DATA --output RUN\nleaklab compare --run RUN --method redact_compress --baseline original --output RUN/comparison.json"
      >
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 18, right: 25, bottom: 28, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e6df" />
            <XAxis
              type="number"
              dataKey="utility"
              name="Task utility"
              unit="%"
              domain={[0, 105]}
              label={{
                value: "Task utility (%) →",
                position: "bottom",
                offset: 10,
              }}
              tick={{ fontSize: 11 }}
            />
            <YAxis
              type="number"
              dataKey="recovery"
              name="Secret recovery"
              unit="%"
              domain={[0, 105]}
              tick={{ fontSize: 11 }}
            />
            <Tooltip />
            <Legend verticalAlign="top" height={35} />
            {names.map((m, i) => (
              <Scatter
                name={label(m)}
                key={m}
                data={points
                  .filter((r) => r.key === m)
                  .sort((a, b) => a.utility - b.utility)}
                fill={colors[i]}
                line
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </Figure>
      <SectionTitle title="Paired comparison with the original context" />
      <p className="section-intro">
        Negative recovery differences mean less recovery. Check the utility and
        length conditions before interpreting the change as an improvement.
      </p>
      <div className="table-scroll panel">
        <table>
          <thead>
            <tr>
              <th>Requested retention</th>
              <th>Recovery difference</th>
              <th>95% interval</th>
              <th>Length matched</th>
              <th>Utility matched</th>
            </tr>
          </thead>
          <tbody>
            {comparisons.map((r, i) => (
              <tr key={i}>
                <td>{pct(Number(r.ratio))}</td>
                <td>{(Number(r.mean) * 100).toFixed(1)} pp</td>
                <td>
                  {(Number(r.low) * 100).toFixed(1)} to{" "}
                  {(Number(r.high) * 100).toFixed(1)} pp
                </td>
                <td>{r.length_match_within_5_percent ? "Yes" : "No"}</td>
                <td>
                  {r.utility_match_within_2_percentage_points ? "Yes" : "No"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Notice>
        These comparisons are exploratory and have no multiplicity correction. A
        lower recovery score with a large loss in task utility is not evidence
        of a better frontier.
      </Notice>
    </>
  );
}

function TheoryResults({ data }: { data: Catalog }) {
  const [budget, setBudget] = useState("0");
  const points = data.privacy.frontier.filter(
    (r) => r.error_budget === Number(budget),
  );
  const feasible = points.filter((r) => r.status === "numerically_certified");
  const regimes = unique(points.map((p) => p.regime));
  const chart = useMemo(
    () =>
      unique(points.map((r) => String(r.length_budget)))
        .map((length) => {
          const row: Record<string, number | string> = {
            length: Number(length),
          };
          for (const regime of regimes) {
            const p = feasible.find(
              (r) => r.regime === regime && r.length_budget === Number(length),
            );
            if (p?.leakage_bits !== undefined) row[regime] = p.leakage_bits;
          }
          return row;
        })
        .sort((a, b) => Number(a.length) - Number(b.length)),
    [budget, data],
  );
  return (
    <>
      <div className="filter-bar">
        <Filter
          name="Maximum task error"
          value={budget}
          onChange={setBudget}
          options={unique(
            data.privacy.frontier.map((p) => String(p.error_budget)),
          ).map((v) => ({ value: v, label: pct(Number(v)) }))}
        />
        <Badge kind="finite_source" />
      </div>
      <Figure
        title="The finite privacy–utility frontier"
        subtitle="Minimum leakage in bits under a fixed decoder and an expected-length budget. Missing points are infeasible."
        evidence="finite_source"
        rows={points.map((p) => ({
          regime: p.regime,
          length_budget: p.length_budget,
          error_budget: p.error_budget,
          status: p.status,
          leakage_bits: p.leakage_bits ?? "Infeasible",
          numerical_gap_bits: p.numerical_gap_bits ?? "Unavailable",
        }))}
        observations={`${feasible.length} of ${points.length} selected grid points have numerical certificates. ${points.length - feasible.length} are infeasible.`}
        interpretation="At zero task error and expected length 0.5, the controls give 0 bits for independent secrets, about 0.531 bits for 90%-correlated secrets, and 1 bit for task-essential secrets."
        limitations="These are floating-point certificates for one finite source and fixed decoder. The calculation is not a privacy guarantee for natural-language compression."
        reproduce="leaklab theory --output frontier.json"
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={chart}
            margin={{ top: 20, right: 25, bottom: 28, left: 8 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e6df" />
            <XAxis
              dataKey="length"
              type="number"
              domain={[0, 1]}
              label={{
                value: "Expected-length budget",
                position: "bottom",
                offset: 10,
              }}
              tick={{ fontSize: 11 }}
            />
            <YAxis
              domain={[0, 1.05]}
              tick={{ fontSize: 11 }}
              label={{
                value: "Leakage (bits)",
                angle: -90,
                position: "insideLeft",
              }}
            />
            <Tooltip />
            <Legend verticalAlign="top" height={35} />
            {regimes.map((r, i) => (
              <Line
                type="linear"
                key={r}
                name={label(r)}
                dataKey={r}
                stroke={colors[i]}
                strokeWidth={2}
                dot={{ r: 4 }}
                connectNulls={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Figure>
    </>
  );
}
