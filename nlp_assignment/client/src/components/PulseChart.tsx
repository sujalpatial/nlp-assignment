import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

interface DriftDataPoint {
  layer: number;
  drift: number;
  timestamp: number;
}

interface PulseChartProps {
  data: DriftDataPoint[];
  className?: string;
}

export default function PulseChart({ data, className = '' }: PulseChartProps) {
  // Transform data for Recharts: group by timestamp
  const chartData = Array.from(
    new Map(
      data.map((d) => [
        d.timestamp,
        { timestamp: d.timestamp, ...data.filter((x) => x.timestamp === d.timestamp).reduce((acc, x) => ({ ...acc, [`layer_${x.layer}`]: x.drift }), {}) },
      ])
    ).values()
  );

  const colors = [
    '#6366f1', // indigo
    '#8b5cf6', // violet
    '#a855f7', // purple
    '#d946ef', // fuchsia
    '#ec4899', // pink
  ];

  return (
    <div className={`w-full h-80 ${className}`}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={chartData}
          margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
          <XAxis
            dataKey="timestamp"
            stroke="var(--color-muted-foreground)"
            style={{ fontSize: '12px' }}
          />
          <YAxis
            stroke="var(--color-muted-foreground)"
            style={{ fontSize: '12px' }}
            label={{ value: 'Drift Score (σ)', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'var(--color-card)',
              border: '1px solid var(--color-border)',
              borderRadius: '8px',
            }}
            labelStyle={{ color: 'var(--color-foreground)' }}
          />
          <Legend />
          <ReferenceLine
            y={3}
            stroke="#dc2626"
            strokeDasharray="5 5"
            label={{
              value: 'Hallucination Threshold (σ=3.0)',
              position: 'right',
              fill: '#dc2626',
              fontSize: 12,
            }}
          />
          {/* Render lines for each layer */}
          {Array.from(new Set(data.map((d) => d.layer)))
            .slice(0, 5)
            .map((layer, idx) => (
              <Line
                key={`layer-${layer}`}
                type="monotone"
                dataKey={`layer_${layer}`}
                stroke={colors[idx % colors.length]}
                dot={false}
                isAnimationActive={true}
                name={`Layer ${layer}`}
                strokeWidth={2}
              />
            ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
