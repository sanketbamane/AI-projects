function SupplierTable({ suppliers }) {

  return (

    <div className="bg-slate-900/80 backdrop-blur-md p-6 rounded-xl border border-slate-800/80 text-white shadow-xl">

      <h2 className="text-xl font-bold mb-4 text-white">

        Supplier Risk Table

      </h2>

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">

          <thead>

            <tr className="border-b border-slate-800 text-slate-400 font-semibold">

              <th className="pb-3 px-2">Name</th>
              <th className="pb-3 px-2">Country</th>
              <th className="pb-3 px-2">Lead Time</th>
              <th className="pb-3 px-2">Risk</th>

            </tr>

          </thead>

          <tbody>

            {suppliers.map(s => (

              <tr
                key={s.id}
                className="border-b border-slate-800/60 hover:bg-slate-800/30 transition-colors"
              >

                <td className="py-3 px-2 text-slate-200">{s.supplier_name}</td>

                <td className="py-3 px-2 text-slate-300">{s.country}</td>

                <td className="py-3 px-2 text-slate-300">{s.lead_time}</td>

                <td className="py-3 px-2">

                  {s.risk_score === 1 ? (

                    <span
                      className="
                      bg-red-500/20
                      border
                      border-red-500/30
                      text-red-400
                      px-2.5
                      py-0.5
                      rounded-full
                      text-xs
                      font-semibold"
                    >
                      HIGH
                    </span>

                  ) : (

                    <span
                      className="
                      bg-green-500/20
                      border
                      border-green-500/30
                      text-green-400
                      px-2.5
                      py-0.5
                      rounded-full
                      text-xs
                      font-semibold"
                    >
                      SAFE
                    </span>

                  )}

                </td>

              </tr>

            ))}

          </tbody>

        </table>
      </div>

    </div>
  );
}

export default SupplierTable;