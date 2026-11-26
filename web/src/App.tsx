import { ExplorePanel, TabularDemo } from "@srikarbolleddu/ml-demo-kit";
import { client, isLive } from "./client";

const CASE_STUDY =
  "https://srikarbolleddu.com/projects/purchase-prediction-system";
const REPO = "https://github.com/srikarbolleddu/ml-purchase-prediction";
const DATASET = "https://archive.ics.uci.edu/dataset/222/bank+marketing";

export function App() {
  return (
    <div className="page">
      <nav className="topbar" aria-label="Site">
        <a href={CASE_STUDY}>← Back to the case study</a>
        <a href={REPO} target="_blank" rel="noopener noreferrer">
          View code <span aria-hidden="true">↗</span>
          <span className="visually-hidden"> (opens in a new tab)</span>
        </a>
      </nav>

      <header className="intro">
        <p className="eyebrow">Live demo · Decision tree</p>
        <h1>Will this customer buy a term deposit?</h1>
        <p>
          A bank called 41,188 customers about a term deposit; about 1 in 9 said
          yes. This pruned decision tree predicts who will buy before the call
          is made, and shows the exact rules behind every prediction. Pick a
          real customer below or change any input.
        </p>
      </header>

      {!isLive && (
        <p className="notice" role="note">
          Showing predictions recorded from the model. The live model API isn't
          connected yet, so edited inputs return the closest recorded example.
        </p>
      )}

      <section aria-labelledby="try" className="section">
        <h2 id="try">Try it</h2>
        <TabularDemo client={client} hideHeader />
      </section>

      <section aria-labelledby="explore" className="section">
        <h2 id="explore">Explore the model</h2>
        <ExplorePanel client={client} />
      </section>

      <footer className="footer">
        <p>
          Capstone project, UT Austin Post-Graduate Program in AI &amp; Machine
          Learning, rebuilt from scratch on an openly licensed dataset.
        </p>
        <p>
          Data:{" "}
          <a href={DATASET} target="_blank" rel="noopener noreferrer">
            UCI Bank Marketing
          </a>{" "}
          (Moro, Rita &amp; Cortez, 2014), CC BY 4.0. Nothing you enter is
          stored.
        </p>
      </footer>
    </div>
  );
}
