import { lazy, Suspense, useEffect, useRef, useState } from "react";
import {
  HashRouter,
  Link,
  NavLink,
  Route,
  Routes,
  useLocation,
} from "react-router-dom";
import {
  ArrowUpRight,
  BookOpen,
  ChartNoAxesCombined,
  Database,
  FlaskConical,
  CodeXml,
  House,
  Layers,
  Menu,
  Play,
  Search,
  ShieldCheck,
  Terminal,
  UserRound,
  X,
} from "lucide-react";
import Overview from "./pages/Overview";
import Methods from "./pages/Methods";
import {
  About,
  Datasets,
  Experiments,
  Reproduce,
  Validation,
} from "./pages/Notebook";
import { resource, useCatalog } from "./data";
import { methods, metrics, testGroups } from "./content";
import type { Literature as LiteratureData } from "./types";
const Literature = lazy(() => import("./pages/Literature"));
const Results = lazy(() => import("./pages/Results"));
const RunCenter = lazy(() => import("./pages/RunCenter"));
const nav = [
  ["/", "Overview", House],
  ["/methods", "Methods", Layers],
  ["/literature", "Literature & evidence", BookOpen],
  ["/datasets", "Datasets & protocols", Database],
  ["/experiments", "Experiments", FlaskConical],
  ["/results", "Results & analysis", ChartNoAxesCombined],
  ["/validation", "Tests & validation", ShieldCheck],
  ["/run", "Run center", Play],
  ["/reproduce", "Reproduce", Terminal],
  ["/about", "About the research", UserRound],
] as const;

function SearchDialog({ close }: { close: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [query, setQuery] = useState("");
  const [literature, setLiterature] = useState<LiteratureData>();
  useEffect(() => {
    dialog.current?.showModal();
    resource<LiteratureData>("literature.json")
      .then(setLiterature)
      .catch(() => {});
    return () => dialog.current?.close();
  }, []);
  const q = query.toLowerCase();
  const found = methods.filter((m) =>
    `${m.name} ${m.description} ${m.tests}`.toLowerCase().includes(q),
  );
  const papers =
    literature?.papers
      .filter((p) => p.title.toLowerCase().includes(q))
      .slice(0, 6) || [];
  return (
    <dialog
      aria-label="Research search"
      ref={dialog}
      className="search-dialog"
      onCancel={close}
    >
      <div className="dialog-heading">
        <Search size={20} />
        <input
          autoFocus
          aria-label="Search research"
          placeholder="Search methods, metrics, tests, and papers…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button aria-label="Close search" onClick={close}>
          <X size={20} />
        </button>
      </div>
      <div className="search-results">
        {query ? (
          <>
            <div className="eyebrow">METHODS & DOCUMENTATION</div>
            {found.map((m) => (
              <Link key={m.id} to={"/methods/" + m.id} onClick={close}>
                {m.name}
                <span>{m.subtitle}</span>
              </Link>
            ))}
            {metrics.some((m) => m.join(" ").toLowerCase().includes(q)) && (
              <Link to="/datasets" onClick={close}>
                Metric glossary
                <span>Definitions, units, and evaluation protocol</span>
              </Link>
            )}
            {testGroups.some((g) =>
              `${g.name} ${g.checks}`.toLowerCase().includes(q),
            ) && (
              <Link to="/validation" onClick={close}>
                Validation and test groups
                <span>What the engineering checks establish</span>
              </Link>
            )}
            <div className="eyebrow">LITERATURE</div>
            {papers.map((p) => (
              <Link
                key={p.id}
                to={"/literature?q=" + encodeURIComponent(p.title)}
                onClick={close}
              >
                {p.title}
                <span>{p.arxiv_id || "Source record"}</span>
              </Link>
            ))}
            <Link
              to={"/literature?q=" + encodeURIComponent(query)}
              onClick={close}
            >
              Search all paper records →
            </Link>
          </>
        ) : (
          <p className="muted">
            Try “attention”, “privacy”, “F1”, or a paper title.
          </p>
        )}
      </div>
    </dialog>
  );
}
function Shell() {
  const { data, error } = useCatalog();
  const [search, setSearch] = useState(false);
  const [mobile, setMobile] = useState(false);
  const location = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
    setMobile(false);
    const name =
      nav.find(
        ([path]) => path !== "/" && location.pathname.startsWith(path),
      )?.[1] || "Overview";
    document.title = `${name} · Prompt Compression Research · Ganea Corneliu`;
  }, [location.pathname]);
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (
        e.key === "/" &&
        !(e.target instanceof HTMLInputElement) &&
        !(e.target instanceof HTMLTextAreaElement)
      ) {
        e.preventDefault();
        setSearch(true);
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, []);
  return (
    <div className="app">
      <a
        className="skip-link"
        href="#main"
        onClick={(e) => {
          e.preventDefault();
          document.getElementById("main")?.focus();
        }}
      >
        Skip to content
      </a>
      <aside className={"sidebar " + (mobile ? "open" : "")}>
        <Link to="/" className="brand">
          <div className="brand-mark">
            <i />
            <i />
            <i />
          </div>
          <div>
            Prompt Compression<span>RESEARCH NOTEBOOK</span>
          </div>
        </Link>
        <div className="nav-label">EXPLORE THE RESEARCH</div>
        <nav>
          {nav.map(([path, name, Icon], i) => (
            <NavLink
              end={path === "/"}
              to={path}
              key={path}
              className={({ isActive }) =>
                (isActive ? "active " : "") + (i === 7 ? "nav-divider" : "")
              }
            >
              <Icon size={17} />
              {name}
              {path === "/run" && <span className="owner-dot" />}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <a
            className="repo-link"
            href="https://github.com/cornelpascal/prompt-compression-research"
            target="_blank"
            rel="noreferrer"
          >
            <CodeXml size={17} />
            Source & reproducibility
            <ArrowUpRight size={13} />
          </a>
          <div className="author">
            <div className="avatar">GC</div>
            <div>
              Ganea Corneliu<span>Research author · 2026</span>
            </div>
          </div>
        </div>
      </aside>
      {mobile && (
        <button
          className="sidebar-overlay"
          aria-label="Close navigation"
          onClick={() => setMobile(false)}
        />
      )}
      <div className="workspace">
        <header className="topbar">
          <div className="breadcrumb">
            <button
              className="mobile-menu"
              aria-label="Open navigation"
              onClick={() => setMobile(!mobile)}
            >
              <Menu size={21} />
            </button>
            <span>Research workspace</span>
            <span>/</span>
            <strong>
              {nav.find(
                ([path]) => path !== "/" && location.pathname.startsWith(path),
              )?.[1] || "Overview"}
            </strong>
          </div>
          <button
            aria-label="Search research"
            className="search-trigger"
            onClick={() => setSearch(true)}
          >
            <Search size={15} />
            <span>Search research</span>
            <kbd>/</kbd>
          </button>
        </header>
        <main id="main" tabIndex={-1}>
          {error ? (
            <div className="error-banner">
              {error}
              <button
                className="button subtle"
                onClick={() => window.location.reload()}
              >
                Reload
              </button>
            </div>
          ) : !data ? (
            <div className="loading">Loading the research notebook…</div>
          ) : (
            <Suspense
              fallback={<div className="loading">Opening this section…</div>}
            >
              <Routes>
                <Route path="/" element={<Overview data={data} />} />
                <Route path="/methods" element={<Methods />} />
                <Route path="/methods/:id" element={<Methods />} />
                <Route path="/literature" element={<Literature />} />
                <Route path="/datasets" element={<Datasets data={data} />} />
                <Route
                  path="/experiments"
                  element={<Experiments data={data} />}
                />
                <Route path="/results" element={<Results data={data} />} />
                <Route
                  path="/validation"
                  element={<Validation data={data} />}
                />
                <Route path="/run" element={<RunCenter />} />
                <Route path="/reproduce" element={<Reproduce />} />
                <Route path="/about" element={<About data={data} />} />
                <Route
                  path="*"
                  element={
                    <div className="empty">
                      <h1>Page not found</h1>
                      <Link to="/">Return to the notebook</Link>
                    </div>
                  }
                />
              </Routes>
            </Suspense>
          )}
          <footer>
            <span>Ganea Corneliu · Prompt Compression Research</span>
            <span>Evidence first. Claims within their limits.</span>
            <a
              href="https://github.com/cornelpascal/prompt-compression-research"
              target="_blank"
              rel="noreferrer"
            >
              View source ↗
            </a>
          </footer>
        </main>
      </div>
      {search && <SearchDialog close={() => setSearch(false)} />}
    </div>
  );
}
export default function App() {
  return (
    <HashRouter>
      <Shell />
    </HashRouter>
  );
}
