import { Link, useParams } from "react-router-dom";
import { ArrowRight, ArrowUpRight } from "lucide-react";
import { methods } from "../content";
import { Badge, Code, PageTitle, SectionTitle, Source } from "../components";

export default function Methods() {
  const { id } = useParams();
  const method = methods.find((m) => m.id === id);
  if (!method)
    return (
      <>
        <PageTitle
          eyebrow="THE RESEARCH TOOLKIT"
          title="Methods, made inspectable."
          description="Six components connect prompt compression, task fidelity, and information recoverability. Each method has a defined input boundary, an evaluation procedure, and explicit limitations."
        />
        <div className="method-grid">
          {methods.map((m) => (
            <Link
              className="method-card panel"
              to={"/methods/" + m.id}
              key={m.id}
            >
              <span className="method-number">{m.number}</span>
              <div className="eyebrow">{m.group}</div>
              <h2>{m.name}</h2>
              <span className="method-subtitle">{m.subtitle}</span>
              <p>{m.description}</p>
              <div className="card-cta">
                Read the method
                <ArrowUpRight size={17} />
              </div>
            </Link>
          ))}
        </div>
      </>
    );
  return (
    <>
      <Link className="back-link" to="/methods">
        ← All methods
      </Link>
      <PageTitle
        eyebrow={method.group}
        title={method.name}
        description={method.description}
      />
      <div className="method-top">
        <Badge
          kind={
            method.id === "theory"
              ? "finite_source"
              : method.group === "Prompt compression"
                ? "engineering_fixture"
                : "synthetic_pilot"
          }
        />
        <Source path={method.source} />
      </div>
      <div className="formula">
        <span>CORE OPERATION</span>
        <code>{method.formula}</code>
      </div>
      <SectionTitle title="How it works" />
      <div className="flow-steps">
        {method.steps.map((s, i) => (
          <div key={s}>
            <span>{String(i + 1).padStart(2, "0")}</span>
            <p>{s}</p>
            {i < 3 && <ArrowRight size={17} />}
          </div>
        ))}
      </div>
      <div className="two-columns">
        <section className="panel prose">
          <h3>Input and output contract</h3>
          <h4>Input</h4>
          <p>{method.inputs}</p>
          <h4>Output</h4>
          <p>{method.outputs}</p>
          <h4>Model or solver</h4>
          <p>{method.model}</p>
        </section>
        <section className="panel prose">
          <h3>Why this approach</h3>
          <p>{method.why}</p>
          <h4>What the tests establish</h4>
          <p>{method.tests}</p>
          <h4>Limits and open questions</h4>
          <p>{method.limits}</p>
        </section>
      </div>
      <section className="panel prose">
        <h3>Reproduce</h3>
        <p>
          Use the pinned configuration and a new or matching run directory.
          Replace uppercase placeholders with your dataset and artifacts.
          GPU-backed methods require the configured model servers.
        </p>
        <Code>{method.command}</Code>
        <div className="inline-links">
          <Source path={method.source}>Read the full protocol</Source>
          <Link className="text-link" to="/validation">
            Inspect test evidence
            <ArrowRight size={14} />
          </Link>
          <Link className="text-link" to="/run">
            Open run center
            <ArrowRight size={14} />
          </Link>
        </div>
      </section>
    </>
  );
}
