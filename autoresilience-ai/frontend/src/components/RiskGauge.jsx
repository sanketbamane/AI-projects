function RiskGauge({ suppliers }) {

  const risky =
    suppliers.filter(
      s => s.risk_score === 1
    ).length;

  const score =
    suppliers.length === 0
      ? 0
      : Math.round(
          (risky / suppliers.length) * 100
        );

  return (
    <div className="bg-slate-900/80 backdrop-blur-md p-6 rounded-xl border border-slate-800/80 text-white shadow-xl">

      <h2 className="text-xl font-bold mb-6 text-white">
        Global Risk Index
      </h2>

      <div className="flex justify-center">

        <div className="relative w-40 h-40">

          <svg
            className="w-40 h-40"
            viewBox="0 0 100 100"
          >
            <circle
              cx="50"
              cy="50"
              r="40"
              stroke="#334155"
              strokeWidth="10"
              fill="none"
            />

            <circle
              cx="50"
              cy="50"
              r="40"
              stroke="#ef4444"
              strokeWidth="10"
              fill="none"
              strokeDasharray="251"
              strokeDashoffset={
                251 - (251 * score) / 100
              }
              transform="rotate(-90 50 50)"
            />
          </svg>

          <div className="absolute inset-0 flex items-center justify-center">

            <span className="text-3xl font-bold text-white">
              {score}%
            </span>

          </div>

        </div>

      </div>

    </div>
  );
}

export default RiskGauge;