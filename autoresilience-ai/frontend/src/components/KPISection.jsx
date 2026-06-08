import {
 FaTruck,
 FaExclamationTriangle,
 FaShieldAlt,
 FaDollarSign
} from "react-icons/fa";

function KPISection({ suppliers }) {

 const total = suppliers.length;

 const risky =
 suppliers.filter(
   s => s.risk_score === 1
 ).length;

 const safe = total - risky;

 return (

<div className="grid md:grid-cols-4 gap-6">

<Card
 icon={<FaTruck />}
 title="Suppliers"
 value={total}
/>

<Card
 icon={<FaShieldAlt />}
 title="Safe"
 value={safe}
/>

<Card
 icon={<FaExclamationTriangle />}
 title="High Risk"
 value={risky}
/>

<Card
 icon={<FaDollarSign />}
 title="Risk Cost"
 value={`$${risky*25000}`}
/>

</div>
 );
}

function Card({icon,title,value}){

 return(
<div className="
bg-gradient-to-r
from-slate-800
to-slate-900
text-white
rounded-xl
p-6
shadow-xl">

<div className="text-3xl">
{icon}
</div>

<h3>{title}</h3>

<p className="text-3xl font-bold">
{value}
</p>

</div>
 )
}

export default KPISection;