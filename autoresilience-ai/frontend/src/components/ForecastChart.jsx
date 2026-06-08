import {
 LineChart,
 Line,
 XAxis,
 YAxis,
 Tooltip,
 ResponsiveContainer
}
from "recharts";

function ForecastChart({
 data
}) {

 return (

  <div className="bg-white p-4 rounded shadow">

   <h2 className="font-bold text-xl mb-4">
    Inventory Forecast
   </h2>

   <ResponsiveContainer
    width="100%"
    height={300}
   >

    <LineChart data={data}>

      <XAxis dataKey="ds" />

      <YAxis />

      <Tooltip />

      <Line dataKey="yhat" />

    </LineChart>

   </ResponsiveContainer>

  </div>

 );
}

export default ForecastChart;