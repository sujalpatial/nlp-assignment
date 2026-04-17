import React from 'react';

interface DriftHeatmapProps {
  text: string;
  driftScores?: number[];
  maxDrift?: number;
  className?: string;
}

export default function DriftHeatmap({
  text,
  driftScores = [],
  maxDrift = 5,
  className = '',
}: DriftHeatmapProps) {
  const tokens = text.split(/\s+/);

  const getHeatmapColor = (drift: number): string => {
    const normalized = Math.min(drift / maxDrift, 1);

    if (normalized < 0.33) {
      // Green (faithful)
      return `rgba(34, 197, 94, ${0.2 + normalized * 0.3})`;
    } else if (normalized < 0.66) {
      // Yellow (warning)
      return `rgba(234, 179, 8, ${0.2 + (normalized - 0.33) * 0.3})`;
    } else {
      // Red (hallucination)
      return `rgba(220, 38, 38, ${0.2 + (normalized - 0.66) * 0.3})`;
    }
  };

  return (
    <div className={`p-4 rounded-lg border border-border bg-card ${className}`}>
      <div className="flex flex-wrap gap-2">
        {tokens.map((token, idx) => {
          const drift = driftScores[idx] || 0;
          const bgColor = getHeatmapColor(drift);

          return (
            <span
              key={idx}
              className="px-2 py-1 rounded text-sm font-mono transition-all duration-200 hover:shadow-md"
              style={{
                backgroundColor: bgColor,
                color: drift > 3 ? '#dc2626' : drift > 2 ? '#ea8f08' : '#22c55e',
              }}
              title={`Drift: ${drift.toFixed(3)}`}
            >
              {token}
            </span>
          );
        })}
      </div>

      {/* Legend */}
      <div className="mt-4 flex items-center gap-4 text-xs">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded" style={{ backgroundColor: 'rgba(34, 197, 94, 0.5)' }} />
          <span className="text-muted-foreground">Faithful</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded" style={{ backgroundColor: 'rgba(234, 179, 8, 0.5)' }} />
          <span className="text-muted-foreground">Warning</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded" style={{ backgroundColor: 'rgba(220, 38, 38, 0.5)' }} />
          <span className="text-muted-foreground">Hallucination</span>
        </div>
      </div>
    </div>
  );
}
