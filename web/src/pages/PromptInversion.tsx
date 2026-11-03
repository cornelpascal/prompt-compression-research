import { useEffect, useState } from "react";
import { Code, Notice, PageTitle, SectionTitle, Source } from "../components";
import { resource } from "../data";

type InversionSummary = {
  recordedAt: string;
  model: string;
  requests: number;
  families: {
    family: string;
    pairs: number;
    controlExactPrompt: number;
    controlMarker: number;
    liveInputTokens: number;
    liveOutputTokens: number;
    liveLatencySeconds: number;
  }[];
};

export default function PromptInversion() {
  const [result, setResult] = useState<InversionSummary>();
  useEffect(() => {
    resource<InversionSummary>("prompt-inversion.json").then(setResult).catch(() => {});
  }, []);
  return (
    <>
      <PageTitle
        eyebrow="BLACK-BOX EXPERIMENT"
        title="Prompt inversion"
        description="Can a model's visible replies disclose the hidden instructions that govern it? This experiment probes GPT-6 Luna and keeps verifiable controls separate from unverified live claims."
      />
      <Notice>
        {result
          ? `${result.requests} API requests were recorded on ${result.recordedAt.slice(0, 10)}. Controlled recovery counts are measurable; Luna's hidden prompt remains unavailable as ground truth.`
          : "No complete GPT-6 Luna result has been published yet. A reply that claims to reveal Luna's system prompt cannot verify its own accuracy."}
      </Notice>
      {result && (
        <section className="panel prose">
          <h2>Measured probe summary</h2>
          <p>Model: {result.model}. Six paired probes per technique. Live token counts and latency include API processing and network time.</p>
          <div className="table-scroll">
            <table>
              <thead><tr><th>Technique</th><th>Control full prompt</th><th>Control marker</th><th>Live input tokens</th><th>Live output tokens</th><th>Live seconds</th></tr></thead>
              <tbody>{result.families.map((item) => (
                <tr key={item.family}>
                  <td>{item.family.replaceAll("_", " ")}</td>
                  <td>{item.controlExactPrompt}/{item.pairs}</td>
                  <td>{item.controlMarker}/{item.pairs}</td>
                  <td>{item.liveInputTokens}</td>
                  <td>{item.liveOutputTokens}</td>
                  <td>{item.liveLatencySeconds}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
          <p>No live response is scored as verified system-prompt recovery.</p>
        </section>
      )}
      <SectionTitle title="What the experiment can establish" />
      <div className="two-columns">
        <section className="panel prose">
          <h3>Known-prompt control</h3>
          <p>
            We supply a synthetic system prompt with a random marker. Exact
            marker and full-prompt recovery can be measured because the
            experiment knows the original text.
          </p>
        </section>
        <section className="panel prose">
          <h3>Live Luna probe</h3>
          <p>
            The same user prompts are sent without a supplied system message.
            We record Luna's visible responses, status, token usage, and
            latency. Any claimed hidden instructions remain unverified.
          </p>
        </section>
      </div>
      <SectionTitle title="Bounded protocol" />
      <section className="panel prose">
        <p>
          Twenty-four fixed probes span direct requests, claimed authority,
          transformations, and transcript continuation. Each probe is paired
          across the control and live target, for at most 48 API requests.
          The model is <strong>gpt-6-luna</strong> with medium reasoning and no
          tools. Only visible output text is observed; internal reasoning and
          next-token probabilities are outside this experiment.
        </p>
        <Code>
          {"cd /Users/pax/Projects/prompt-compression-research\n.venv/bin/python research/prompt_compression/secure_openai_config.py init\n.venv/bin/python research/prompt_compression/prompt_inversion.py --output research/prompt_compression/runs/inversion-smoke --limit 2\n.venv/bin/python research/prompt_compression/prompt_inversion.py --output research/prompt_compression/runs/inversion-full --limit 48"}
        </Code>
        <p>
          The encrypted key and raw run files stay outside Git. Measured
          summaries and carefully labelled examples can be published here
          after the run is reviewed.
        </p>
        <Source path="research/prompt_compression/prompt_inversion.py">
          Probe runner and fixed attack set
        </Source>
      </section>
    </>
  );
}
