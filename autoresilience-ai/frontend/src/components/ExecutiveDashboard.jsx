function ExecutiveDashboard({
 suppliers
}) {

 const total =
 suppliers.length;

 const risky =
 suppliers.filter(
 x=>x.risk_score===1
 ).length;

 const safe =
 total-risky;

 return (

 <div
 className="
 bg-gradient-to-r
 from-blue-600
 to-indigo-700
 text-white
 p-6
 rounded
 shadow"
 >

 <h2
 className="
 text-3xl
 font-bold"
 >

 Executive Summary

 </h2>

 <div
 className="
 grid
 grid-cols-3
 gap-4
 mt-4"
 >

 <div>

 Total Suppliers

 <h3 className="text-4xl">

 {total}

 </h3>

 </div>

 <div>

 High Risk

 <h3 className="text-4xl">

 {risky}

 </h3>

 </div>

 <div>

 Safe

 <h3 className="text-4xl">

 {safe}

 </h3>

 </div>

 </div>

 </div>

 );
}

export default ExecutiveDashboard;