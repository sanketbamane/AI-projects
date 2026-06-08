import {
 PieChart,
 Pie,
 Cell,
 Tooltip
} from "recharts";

function RiskDonutChart({ suppliers }) {

 const risky =
 suppliers.filter(
  s=>s.risk_score===1
 ).length;

 const safe =
 suppliers.length-risky;

 const data=[
  {name:"Safe",value:safe},
  {name:"Risky",value:risky}
 ];

 return(

 <div className="bg-slate-900/80 backdrop-blur-md p-6 rounded-xl border border-slate-800/80 text-white shadow-xl">

 <h2 className="font-bold mb-4 text-white">
 Risk Distribution
 </h2>

 <PieChart width={350} height={300}>

 <Pie
  data={data}
  innerRadius={60}
  outerRadius={100}
  dataKey="value"
 >

 <Cell fill="#22c55e"/>
 <Cell fill="#ef4444"/>

 </Pie>

 <Tooltip
   contentStyle={{
     backgroundColor: '#1e293b',
     borderColor: '#334155',
     borderRadius: '8px',
     color: '#fff'
   }}
   itemStyle={{ color: '#fff' }}
 />

 </PieChart>

 </div>

 )

}

export default RiskDonutChart;