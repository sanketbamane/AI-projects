import { useState } from "react";
import API from "../services/api";

function SupplierForm({ refresh }) {

  const [form, setForm] = useState({

    supplier_name: "",
    country: "",
    lead_time: 0,
    on_time_delivery: 0,
    defect_rate: 0,
    cost_score: 0

  });

  const submit = async () => {

    await API.post(
      "/suppliers",
      form
    );

    refresh();

    alert("Supplier Added");
  };

  return (

    <div className="bg-slate-900/80 backdrop-blur-md p-6 rounded-xl border border-slate-800/80 text-white shadow-xl">

      <h2 className="font-bold text-xl mb-4 text-white">
        Add Supplier
      </h2>

      <input
        placeholder="Supplier Name"
        className="bg-slate-800 border border-slate-700/80 p-2 rounded w-full mb-3 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
        onChange={e =>
          setForm({
            ...form,
            supplier_name:e.target.value
          })
        }
      />

      <input
        placeholder="Country"
        className="bg-slate-800 border border-slate-700/80 p-2 rounded w-full mb-3 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
        onChange={e =>
          setForm({
            ...form,
            country:e.target.value
          })
        }
      />

      <input
        placeholder="Lead Time"
        className="bg-slate-800 border border-slate-700/80 p-2 rounded w-full mb-3 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
        type="number"
        onChange={e =>
          setForm({
            ...form,
            lead_time:Number(e.target.value)
          })
        }
      />

      <input
        placeholder="On-Time Delivery"
        className="bg-slate-800 border border-slate-700/80 p-2 rounded w-full mb-3 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
        type="number"
        onChange={e =>
          setForm({
            ...form,
            on_time_delivery:Number(e.target.value)
          })
        }
      />

      <input
        placeholder="Defect Rate"
        className="bg-slate-800 border border-slate-700/80 p-2 rounded w-full mb-3 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
        type="number"
        onChange={e =>
          setForm({
            ...form,
            defect_rate:Number(e.target.value)
          })
        }
      />

      <input
        placeholder="Cost Score"
        className="bg-slate-800 border border-slate-700/80 p-2 rounded w-full mb-3 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
        type="number"
        onChange={e =>
          setForm({
            ...form,
            cost_score:Number(e.target.value)
          })
        }
      />

      <button
        onClick={submit}
        className="
        bg-blue-600
        hover:bg-blue-500
        active:bg-blue-700
        text-white
        font-semibold
        px-4
        py-2
        rounded-lg
        transition-colors
        duration-200
        shadow-md
        shadow-blue-900/20"
      >
        Predict & Save
      </button>

    </div>
  );
}

export default SupplierForm;