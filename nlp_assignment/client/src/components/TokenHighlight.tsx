import React from 'react';
import { AlertCircle, AlertTriangle } from 'lucide-react';

interface Token {
  text: string;
  drift_score: number;
  is_hallucination: boolean;
  entropy: number;
}

interface TokenHighlightProps {
  tokens: Token[];
  className?: string;
}

export default function TokenHighlight({ tokens, className = '' }: TokenHighlightProps) {
  const getTokenColor = (token: Token): string => {
    if (token.is_hallucination) {
      return 'bg-danger/20 text-danger border border-danger/30';
    }
    if (token.drift_score > 2.0) {
      return 'bg-warning/20 text-warning border border-warning/30';
    }
    return 'bg-success/20 text-success border border-success/30';
  };

  const getTokenIcon = (token: Token) => {
    if (token.is_hallucination) {
      return <AlertCircle className="w-3 h-3 inline mr-1" />;
    }
    if (token.drift_score > 2.0) {
      return <AlertTriangle className="w-3 h-3 inline mr-1" />;
    }
    return null;
  };

  return (
    <div className={`flex flex-wrap gap-2 p-4 rounded-lg border border-border bg-card ${className}`}>
      {tokens.length === 0 ? (
        <p className="text-sm text-muted-foreground">No tokens generated yet</p>
      ) : (
        tokens.map((token, idx) => (
          <div
            key={idx}
            className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium transition-all duration-200 hover:shadow-md ${getTokenColor(
              token
            )}`}
            title={`Drift: ${token.drift_score.toFixed(3)}, Entropy: ${token.entropy.toFixed(3)}`}
          >
            {getTokenIcon(token)}
            <span className="font-mono">{token.text}</span>
            {token.is_hallucination && (
              <span className="ml-1 text-xs font-semibold">Potential Hallucination</span>
            )}
          </div>
        ))
      )}
    </div>
  );
}
