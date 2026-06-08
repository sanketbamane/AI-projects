import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer
} from "recharts";

function RiskChart({ suppliers }) {

  return (

    <div className="bg-slate-900/80 backdrop-blur-md p-6 rounded-xl border border-slate-800/80 text-white shadow-xl">

      <h2 className="text-xl font-bold mb-4 text-white">
        Supplier Risks
      </h2>

      <ResponsiveContainer
        width="100%"
        height={300}
      >

        <BarChart data={suppliers}>

          <XAxis
            dataKey="supplier_name"
            stroke="#94a3b8"
          />

          <YAxis stroke="#94a3b8" />

          <Tooltip
            contentStyle={{
              backgroundColor: '#1e293b',
              borderColor: '#334155',
              borderRadius: '8px',
              color: '#fff'
            }}
            itemStyle={{ color: '#fff' }}
          />

          <Bar
            dataKey="risk_score"
            fill="#3b82f6"
            radius={[4, 4, 0, 0]}
          />

        </BarChart>

      </ResponsiveContainer>

    </div>
  );
}

export default RiskChart;