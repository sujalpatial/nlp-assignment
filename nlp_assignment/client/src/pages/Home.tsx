import { Button } from '@/components/ui/button'
import { Zap, Activity, TrendingUp } from 'lucide-react'
import { useLocation } from 'wouter'

export default function Home() {
  const [, navigate] = useLocation()

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted">
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap className="w-6 h-6 text-accent" />
            <h1 className="text-2xl font-bold text-foreground">Project Sentinel</h1>
          </div>
          <Button onClick={() => navigate('/dashboard')} className="bg-accent hover:bg-accent/90">
            Open Dashboard
          </Button>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-20 space-y-16">
        <div className="text-center space-y-6">
          <h2 className="text-6xl font-bold text-foreground leading-tight">
            Real-Time Hallucination Detection
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
            Mechanistic interpretability meets causal intervention. Detect and suppress
            hallucinations in real-time using transformer layer analysis and PCA-based steering.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          <div className="p-8 rounded-lg border border-border bg-card hover:shadow-lg hover:border-accent/50 transition-all duration-300">
            <div className="flex items-center gap-3 mb-4">
              <Activity className="w-8 h-8 text-accent" />
              <h3 className="font-semibold text-lg">The Pulse</h3>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Real-time visualization of activation drift across all 24 transformer layers
              as tokens are generated. Watch the model's internal state evolve.
            </p>
          </div>
          <div className="p-8 rounded-lg border border-border bg-card hover:shadow-lg hover:border-yellow-500/50 transition-all duration-300">
            <div className="flex items-center gap-3 mb-4">
              <TrendingUp className="w-8 h-8 text-yellow-500" />
              <h3 className="font-semibold text-lg">Token Prediction</h3>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Color-coded token highlighting: Yellow for warnings, Red for potential
              hallucinations (σ &gt; 3.0). Predict problems before they appear.
            </p>
          </div>
          <div className="p-8 rounded-lg border border-border bg-card hover:shadow-lg hover:border-green-500/50 transition-all duration-300">
            <div className="flex items-center gap-3 mb-4">
              <Zap className="w-8 h-8 text-green-500" />
              <h3 className="font-semibold text-lg">Causal Switch</h3>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              PCA-based steering projects hallucinated activations back to the faithful
              subspace. Mechanistically correct the model before sampling.
            </p>
          </div>
        </div>

        <div className="text-center py-8">
          <Button
            size="lg"
            className="bg-accent hover:bg-accent/90 text-white px-10 py-7 text-lg font-semibold shadow-lg"
            onClick={() => navigate('/dashboard')}
          >
            <Zap className="w-5 h-5 mr-2" />
            Launch Dashboard
          </Button>
        </div>

        <div className="mt-20 pt-12 border-t border-border text-center text-sm text-muted-foreground">
          <p>Project Sentinel — CS F429 NLP Research Assignment · Track B · BITS Pilani Dubai</p>
        </div>
      </main>
    </div>
  )
}
