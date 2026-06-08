function AIInsights({ suppliers }) {

 const risky =
 suppliers.filter(
 s=>s.risk_score===1
 );

 return(

<div className="
bg-slate-900
text-white
rounded-xl
p-6
shadow-xl">

<h2 className="font-bold text-xl">

AI Recommendations

</h2>

<ul className="mt-4 space-y-2">

<li>
⚠ {risky.length}
 suppliers require review
</li>

<li>
📦 Recommend inventory buffer
 for risky suppliers
</li>

<li>
🌎 Suggest alternate sourcing
 from India/Vietnam
</li>

<li>
💰 Estimated savings:
 $120,000
</li>

</ul>

</div>

 )

}

export default AIInsights;