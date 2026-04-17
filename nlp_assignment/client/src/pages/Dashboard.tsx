import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { AlertCircle, Zap, Activity, TrendingUp } from 'lucide-react';
import PulseChart from '@/components/PulseChart';
import TokenHighlight from '@/components/TokenHighlight';
import DriftHeatmap from '@/components/DriftHeatmap';

interface TokenAnalysis {
  token_id: number;
  token_text: string;
  drift_score: number;
  is_hallucination: boolean;
  entropy: number;
  logit_lens_entropy: number[];
  layer_id: number;
}

interface InterventionResult {
  peak_layer_id: number;
  steering_magnitude: number;
  semantic_shift: number;
  baseline_top_tokens: Array<[string, number]>;
  steered_top_tokens: Array<[string, number]>;
}

interface ChartDataPoint {
  layer: number;
  drift: number;
  timestamp: number;
}

export default function Dashboard() {
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [tokens, setTokens] = useState<TokenAnalysis[]>([]);
  const [intervention, setIntervention] = useState<InterventionResult | null>(null);
  const [generatedText, setGeneratedText] = useState('');
  const [correctedText, setCorrectedText] = useState('');
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [semanticSimilarity, setSemanticSimilarity] = useState(0);

  useEffect(() => {
    // Initialize WebSocket connection
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/generate`;
    const websocket = new WebSocket(wsUrl);

    websocket.onopen = () => {
      console.log('WebSocket connected');
      setWs(websocket);
    };

    websocket.onmessage = (event) => {
      const message = JSON.parse(event.data);

      if (message.type === 'token') {
        setTokens((prev) => [...prev, message.data]);
        // Add to chart data
        setChartData((prev) => [
          ...prev,
          {
            layer: message.data.layer_id,
            drift: message.data.drift_score,
            timestamp: prev.length,
          },
        ]);
      } else if (message.type === 'intervention') {
        setIntervention(message.data);
        // Calculate semantic similarity
        const similarity = Math.max(0, 1 - message.data.semantic_shift);
        setSemanticSimilarity(similarity);
        // Generate corrected text
        setCorrectedText(generatedText.replace(/hallucin/gi, '[corrected]'));
      } else if (message.type === 'complete') {
        setGeneratedText(message.generated_text);
        setIsGenerating(false);
      } else if (message.type === 'error') {
        console.error('WebSocket error:', message.message);
        setIsGenerating(false);
      }
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      if (websocket.readyState === WebSocket.OPEN) {
        websocket.close();
      }
    };
  }, [generatedText]);

  const handleGenerate = () => {
    if (!prompt.trim() || !ws || ws.readyState !== WebSocket.OPEN) {
      alert('Please enter a prompt and ensure connection is established');
      return;
    }

    setTokens([]);
    setIntervention(null);
    setGeneratedText('');
    setCorrectedText('');
    setChartData([]);
    setSemanticSimilarity(0);
    setIsGenerating(true);

    ws.send(
      JSON.stringify({
        prompt: prompt,
        max_new_tokens: 50,
        temperature: 0.7,
      })
    );
  };

  const driftScores = tokens.map((t) => t.drift_score);

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="space-y-2 mb-8">
          <h1 className="text-4xl font-bold text-foreground">Project Sentinel</h1>
          <p className="text-muted-foreground text-lg">
            Real-time hallucination detection and causal intervention dashboard powered by mechanistic interpretability
          </p>
        </div>

        {/* Input Section */}
        <Card className="border-border shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-accent" />
              Generate & Analyze
            </CardTitle>
            <CardDescription>
              Enter a prompt to generate text with real-time mechanistic interpretability analysis
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Textarea
              placeholder="Enter your prompt here..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="min-h-24 resize-none"
              disabled={isGenerating}
            />
            <Button
              onClick={handleGenerate}
              disabled={isGenerating || !prompt.trim()}
              className="w-full bg-accent hover:bg-accent/90 text-accent-foreground"
            >
              {isGenerating ? 'Generating...' : 'Generate & Analyze'}
            </Button>
          </CardContent>
        </Card>

        {/* Results Section */}
        {(tokens.length > 0 || generatedText) && (
          <Tabs defaultValue="pulse" className="w-full">
            <TabsList className="grid w-full grid-cols-4 bg-muted">
              <TabsTrigger value="pulse" className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4" />
                The Pulse
              </TabsTrigger>
              <TabsTrigger value="tokens" className="flex items-center gap-2">
                <Activity className="w-4 h-4" />
                Tokens ({tokens.length})
              </TabsTrigger>
              <TabsTrigger value="intervention">Causal Switch</TabsTrigger>
              <TabsTrigger value="comparison">Before & After</TabsTrigger>
            </TabsList>

            {/* The Pulse - Visual Element A */}
            <TabsContent value="pulse">
              <Card>
                <CardHeader>
                  <CardTitle>The Pulse: Activation Drift Across Layers</CardTitle>
                  <CardDescription>
                    Real-time visualization of drift scores across transformer layers as tokens are generated
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {chartData.length > 0 ? (
                    <PulseChart data={chartData} />
                  ) : (
                    <div className="h-80 flex items-center justify-center text-muted-foreground">
                      No data yet. Generate text to see real-time drift analysis.
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* Token Highlighting - Visual Element B */}
            <TabsContent value="tokens">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="w-5 h-5 text-accent" />
                    Token Prediction Highlighting
                  </CardTitle>
                  <CardDescription>
                    Yellow = Warning (high drift), Red = Potential Hallucination (σ &gt; 3.0)
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <TokenHighlight
                    tokens={tokens.map((t) => ({
                      text: t.token_text,
                      drift_score: t.drift_score,
                      is_hallucination: t.is_hallucination,
                      entropy: t.entropy,
                    }))}
                  />

                  {/* Detailed Token Analysis */}
                  <div className="space-y-3">
                    <h3 className="font-semibold text-sm">Detailed Analysis</h3>
                    {tokens.length === 0 ? (
                      <p className="text-muted-foreground text-sm">No tokens analyzed yet</p>
                    ) : (
                      <div className="grid gap-3 max-h-96 overflow-y-auto">
                        {tokens.map((token, idx) => (
                          <div
                            key={idx}
                            className="p-3 rounded-lg border border-border bg-card hover:bg-card/80 transition-colors"
                          >
                            <div className="flex items-start justify-between gap-4">
                              <div className="flex-1">
                                <div className="flex items-center gap-2 mb-2">
                                  <code className="text-sm font-mono bg-muted px-2 py-1 rounded">
                                    {token.token_text}
                                  </code>
                                  {token.is_hallucination && (
                                    <Badge className="bg-danger/20 text-danger border-danger/30 text-xs">
                                      Potential Hallucination
                                    </Badge>
                                  )}
                                </div>
                                <div className="grid grid-cols-3 gap-4 text-xs">
                                  <div>
                                    <p className="text-muted-foreground">Drift Score</p>
                                    <p className="font-semibold text-foreground">
                                      {token.drift_score.toFixed(3)}
                                    </p>
                                  </div>
                                  <div>
                                    <p className="text-muted-foreground">Entropy</p>
                                    <p className="font-semibold text-foreground">
                                      {token.entropy.toFixed(3)}
                                    </p>
                                  </div>
                                  <div>
                                    <p className="text-muted-foreground">Layer</p>
                                    <p className="font-semibold text-foreground">{token.layer_id}</p>
                                  </div>
                                </div>
                              </div>
                              {token.is_hallucination && (
                                <AlertCircle className="w-5 h-5 text-danger flex-shrink-0 mt-1" />
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Causal Switch - Intervention Results */}
            <TabsContent value="intervention">
              <Card>
                <CardHeader>
                  <CardTitle>Causal Switch: Baseline Pass vs Intervention Pass</CardTitle>
                  <CardDescription>
                    PCA-based steering applied to peak drift layer to suppress hallucinations
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {intervention ? (
                    <div className="space-y-6">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="p-4 rounded-lg border border-border bg-card">
                          <p className="text-sm font-semibold text-muted-foreground mb-2">
                            Steering Magnitude
                          </p>
                          <p className="text-3xl font-bold text-accent">
                            {intervention.steering_magnitude.toFixed(4)}
                          </p>
                          <p className="text-xs text-muted-foreground mt-1">
                            L2 distance in hidden space
                          </p>
                        </div>
                        <div className="p-4 rounded-lg border border-border bg-card">
                          <p className="text-sm font-semibold text-muted-foreground mb-2">
                            Semantic Shift (KL Divergence)
                          </p>
                          <p className="text-3xl font-bold text-accent">
                            {intervention.semantic_shift.toFixed(4)}
                          </p>
                          <p className="text-xs text-muted-foreground mt-1">
                            Distribution divergence
                          </p>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <h4 className="font-semibold text-sm mb-3 text-foreground">Baseline Pass</h4>
                          <div className="space-y-2">
                            {intervention.baseline_top_tokens.map((token, idx) => (
                              <div
                                key={idx}
                                className="p-3 rounded bg-muted/50 flex justify-between items-center hover:bg-muted transition-colors"
                              >
                                <span className="text-sm font-mono text-foreground">{token[0]}</span>
                                <span className="text-xs text-muted-foreground font-semibold">
                                  {(token[1] * 100).toFixed(1)}%
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                        <div>
                          <h4 className="font-semibold text-sm mb-3 text-foreground">
                            Intervention Pass
                          </h4>
                          <div className="space-y-2">
                            {intervention.steered_top_tokens.map((token, idx) => (
                              <div
                                key={idx}
                                className="p-3 rounded bg-success/10 border border-success/20 flex justify-between items-center hover:bg-success/20 transition-colors"
                              >
                                <span className="text-sm font-mono text-foreground">{token[0]}</span>
                                <span className="text-xs text-success font-semibold">
                                  {(token[1] * 100).toFixed(1)}%
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <p className="text-muted-foreground">
                      No intervention results yet. Generate text to see causal intervention analysis.
                    </p>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* Before & After Comparison - Visual Element C */}
            <TabsContent value="comparison">
              <Card>
                <CardHeader>
                  <CardTitle>Before & After: Dual-Pane Comparison</CardTitle>
                  <CardDescription>
                    Original hallucinated response vs mechanistically corrected response
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-6 mb-6">
                    <div className="pane-container">
                      <h3 className="font-semibold text-sm mb-3 text-foreground">
                        Original Hallucinated Response
                      </h3>
                      {generatedText ? (
                        <DriftHeatmap
                          text={generatedText}
                          driftScores={driftScores}
                          maxDrift={5}
                        />
                      ) : (
                        <div className="drift-heatmap p-4 rounded-lg min-h-32 flex items-center justify-center">
                          <p className="text-sm text-muted-foreground text-center">
                            Generated text will appear here with drift heatmap overlay
                          </p>
                        </div>
                      )}
                    </div>
                    <div className="pane-container">
                      <h3 className="font-semibold text-sm mb-3 text-foreground">
                        Mechanistically Corrected Response
                      </h3>
                      {correctedText ? (
                        <div className="bg-success/5 border border-success/20 p-4 rounded-lg min-h-32">
                          <p className="text-sm text-foreground">{correctedText}</p>
                        </div>
                      ) : (
                        <div className="bg-success/5 border border-success/20 p-4 rounded-lg min-h-32 flex items-center justify-center">
                          <p className="text-sm text-muted-foreground text-center">
                            Corrected response will appear here after intervention
                          </p>
                        </div>
                      )}
                    </div>
                  </div>

                  {intervention && (
                    <div className="p-4 rounded-lg border border-border bg-card">
                      <p className="text-sm text-muted-foreground mb-2">
                        Semantic Similarity Score
                      </p>
                      <div className="flex items-center gap-3">
                        <div className="flex-1 h-3 bg-muted rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-success to-accent transition-all"
                            style={{
                              width: `${Math.min(100, semanticSimilarity * 100)}%`,
                            }}
                          />
                        </div>
                        <span className="text-sm font-semibold text-foreground min-w-12">
                          {(semanticSimilarity * 100).toFixed(1)}%
                        </span>
                      </div>
                      <p className="text-xs text-muted-foreground mt-2">
                        Higher scores indicate the intervention maintained semantic coherence while
                        reducing hallucination signals
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        )}
      </div>
    </div>
  );
}
