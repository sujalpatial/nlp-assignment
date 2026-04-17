import { useAuth } from "@/_core/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { Zap, Activity, TrendingUp } from "lucide-react";
import { getLoginUrl } from "@/const";
import { Streamdown } from 'streamdown';
import { useLocation } from 'wouter';

export default function Home() {
  const [, navigate] = useLocation();
  let { user, isAuthenticated, logout } = useAuth();

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap className="w-6 h-6 text-accent" />
            <h1 className="text-2xl font-bold text-foreground">Project Sentinel</h1>
          </div>
          <div className="flex items-center gap-4">
            {isAuthenticated ? (
              <>
                <span className="text-sm text-muted-foreground">Welcome, {user?.name}</span>
                <Button variant="ghost" onClick={() => logout()}>
                  Logout
                </Button>
              </>
            ) : (
              <Button onClick={() => (window.location.href = getLoginUrl())}>
                Login
              </Button>
            )}
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <main className="max-w-7xl mx-auto px-6 py-20 space-y-16">
        <div className="text-center space-y-6 mb-16">
          <h2 className="text-6xl font-bold text-foreground leading-tight">
            Real-Time Hallucination Detection
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
            Mechanistic interpretability meets causal intervention. Detect and suppress hallucinations in real-time
            using transformer layer analysis and PCA-based steering.
          </p>
        </div>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-3 gap-6 mb-16">
          <div className="p-8 rounded-lg border border-border bg-card hover:shadow-lg hover:border-accent/50 transition-all duration-300">
            <div className="flex items-center gap-3 mb-4">
              <Activity className="w-8 h-8 text-accent" />
              <h3 className="font-semibold text-lg">The Pulse</h3>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Real-time visualization of activation drift across transformer layers as tokens are generated. Watch the model's internal state evolve.
            </p>
          </div>
          <div className="p-8 rounded-lg border border-border bg-card hover:shadow-lg hover:border-warning/50 transition-all duration-300">
            <div className="flex items-center gap-3 mb-4">
              <TrendingUp className="w-8 h-8 text-warning" />
              <h3 className="font-semibold text-lg">Token Prediction</h3>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Color-coded token highlighting: Yellow for warnings, Red for potential hallucinations (σ &gt; 3.0). Predict problems before they happen.
            </p>
          </div>
          <div className="p-8 rounded-lg border border-border bg-card hover:shadow-lg hover:border-success/50 transition-all duration-300">
            <div className="flex items-center gap-3 mb-4">
              <Zap className="w-8 h-8 text-success" />
              <h3 className="font-semibold text-lg">Causal Switch</h3>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              PCA-based steering to project hallucinated activations back to the faithful subspace. Intervene before sampling.
            </p>
          </div>
        </div>

        {/* CTA Button */}
        <div className="text-center py-8">
          <Button
            size="lg"
            className="bg-accent hover:bg-accent/90 text-accent-foreground px-10 py-7 text-lg font-semibold shadow-lg hover:shadow-xl transition-all duration-300"
            onClick={() => navigate('/dashboard')}
          >
            <Zap className="w-5 h-5 mr-2" />
            Launch Dashboard
          </Button>
        </div>

        {/* Technical Details */}
        <div className="mt-20 p-10 rounded-lg border border-border bg-card/50 backdrop-blur-sm space-y-6">
          <h3 className="text-2xl font-semibold text-foreground">How It Works</h3>
          <Streamdown>
{`
## Mechanistic Interpretability Pipeline

### 1. Forward-Hook Extraction
Hidden states h_{l,t} are extracted from each transformer layer during token generation using PyTorch forward hooks. This captures the model's internal representations for every layer and every generated token.

### 2. Drift Scoring
Mahalanobis distance is computed against a pre-computed Faithful Manifold using Ledoit-Wolf shrinkage covariance estimation. This robust statistical measure identifies when the model's activations deviate from patterns seen during faithful generation.

### 3. Hallucination Flagging
Tokens with drift σ > 3.0 are flagged as "Potential Hallucination" BEFORE sampling. This early detection allows intervention before the model commits to generating false information.

### 4. Causal Intervention
The peak drift layer is identified and PCA steering projects activations onto the faithful subspace. This mechanistically corrects the model's internal state to suppress hallucination signals.

### 5. Real-Time Streaming
Per-token analysis streams via WebSocket for sub-100ms latency. The frontend receives real-time updates as the model generates each token.

### 6. Dual-Pane Comparison
Original vs mechanistically corrected responses are displayed side-by-side with semantic similarity scoring. Quantify how much the intervention improved factual grounding.
`}
          </Streamdown>
        </div>

        {/* Architecture Overview */}
        <div className="grid md:grid-cols-2 gap-8 mt-16">
          <div className="p-8 rounded-lg border border-border bg-card space-y-4">
            <h4 className="font-semibold text-lg text-foreground">Backend Architecture</h4>
            <ul className="space-y-3 text-sm text-muted-foreground">
              <li className="flex items-start gap-3">
                <span className="text-accent font-bold">→</span>
                <span>PyTorch inference engine with OPT-1.3b model</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-accent font-bold">→</span>
                <span>Forward hooks on all 24 transformer layers</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-accent font-bold">→</span>
                <span>Ledoit-Wolf shrinkage for robust covariance estimation</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-accent font-bold">→</span>
                <span>FastAPI backend with WebSocket streaming</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-accent font-bold">→</span>
                <span>Real-time per-token analysis and causal intervention</span>
              </li>
            </ul>
          </div>
          <div className="p-8 rounded-lg border border-border bg-card space-y-4">
            <h4 className="font-semibold text-lg text-foreground">Frontend Architecture</h4>
            <ul className="space-y-3 text-sm text-muted-foreground">
              <li className="flex items-start gap-3">
                <span className="text-accent font-bold">→</span>
                <span>React 19 with TypeScript for type safety</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-accent font-bold">→</span>
                <span>Recharts for real-time drift visualization</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-accent font-bold">→</span>
                <span>Tailwind CSS 4 with OKLCH color space</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-accent font-bold">→</span>
                <span>WebSocket client for real-time streaming</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-accent font-bold">→</span>
                <span>Elegant UI with refined typography and micro-interactions</span>
              </li>
            </ul>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-20 pt-12 border-t border-border text-center text-sm text-muted-foreground">
          <p>Project Sentinel: Mechanistic Interpretability for Real-Time Hallucination Detection</p>
          <p className="mt-2">Built with PyTorch, FastAPI, React, and Tailwind CSS</p>
        </div>
      </main>
    </div>
  );
}
