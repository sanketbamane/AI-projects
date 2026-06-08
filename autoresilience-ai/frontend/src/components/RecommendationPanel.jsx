import { useState } from "react";
import API from "../services/api";

function RecommendationPanel() {

 const [id,setId] = useState("");

 const [data,setData] =
 useState([]);

 const search = async ()=>{

   const res =
   await API.get(
   `/recommend/${id}`
   );

   setData(res.data);
 };

 return (

 <div
 className="
 bg-white
 p-4
 rounded
 shadow"
 >

 <h2
 className="
 text-xl
 font-bold
 mb-4"
 >

 Alternate Suppliers

 </h2>

 <input
 placeholder="Supplier ID"
 className="
 border
 p-2
 w-full"
 onChange={(e)=>
 setId(e.target.value)}
 />

 <button
 onClick={search}
 className="
 bg-blue-600
 text-white
 px-4
 py-2
 mt-2
 rounded"
 >

 Search

 </button>

 {
 data.map(item=>(

 <div
 key={item.id}
 className="border p-2 mt-2"
 >

 {item.supplier_name}

 </div>

 ))
 }

 </div>

 );
}

export default RecommendationPanel;